import pygame
import neat  # type: ignore
import os
import sys
from typing import Literal
import random
#from typing import Any
import visualize # from examples
import pathlib
import pickle
import math
import time

pygame.init()

SIZE: tuple[int, int] = (800, 600)
global game_speed, generation_number    # Um sie innerhalb von Funktionen zu nutzen
game_speed: float = 1
generation_number: int = 0
global default_game_speed
default_game_speed: float = 1

SCREEN: pygame.display = pygame.display.set_mode(SIZE) # type: ignore
SCROLL_SPEED: int = 10
p: float = 0.85 # Sicherheit fürs Auslösen von jump (probability)


class Cactus:
    def __init__(self):
        self.height: int = 80
        self.width: int = 40
        self.x: int = SIZE[0]
        self.y: int = SIZE[1] - self.height
        self.distance: int = self.width * 3
        self.color: tuple[int, int, int] = (34, 116, 66)
        self.rect: pygame.rect = pygame.Rect(self.x, self.y, self.width, self.height)


    def set_size(self):
        self.width = random.randint(1, 3) * self.width
        if random.randint(1, 2) == 2:
            self.height = int(1.5 * self.height)
            self.y = SIZE[1] - self.height
        else:
            self.height = int(self.height)
        return self.width, self.height


    def draw(self):
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(SCREEN, self.color, self.rect)

    def update(self, gms: int):
        new_scroll_speed: int = int(SCROLL_SPEED*gms)
        self.x -= new_scroll_speed


    def is_off_screen(self):
        if self.x < -160: # random X value, kein bock zu kalkulieren
            return True
        else:
            return False




class Player:
    def __init__(self):
        self.width: float = 1
        self.height: float = 2
        self.size_mul: int = 20
        self.color: tuple[int, int, int] = (255, 61, 83)
        self.x: int = 50
        self.y: int = int(SIZE[1] - self.height * self.size_mul)
        self.velocity: float = 0
        self.screen = SCREEN
        self.rect: pygame.rect = pygame.Rect(self.x, self.y, self.width * self.size_mul, self.height * self.size_mul)
        self.can_jump: bool = True

    def on_floor(self) -> bool:
        floor_level = SIZE[1] - self.height * self.size_mul
        return self.y >= floor_level


    def gravity(self, gravity: int=1, floor_y: int=SIZE[1]):
        """Apply gravity to player's position."""
        self.velocity += gravity
        self.y += self.velocity

        floor_level = int(floor_y - self.height * self.size_mul)
        if self.y >= floor_level:
            self.y = floor_level
            self.velocity = 0
            self.can_jump = True

    def jump(self, level: int) -> bool:
        level = max(1, min(10, level))

        multiplier = 0.25 * level

        if self.on_floor() and self.can_jump:
            self.velocity -= multiplier * 20
            self.can_jump = False
            return True

        return False

    def draw(self):
        self.rect = pygame.Rect(self.x, self.y, self.width * self.size_mul, self.height * self.size_mul)
        pygame.draw.rect(self.screen, self.color, self.rect)


    def get_next_cactus(self, cacti: list[Cactus]):
        """
                Returns the nearest cactus in front of the player.
                If no cactus is ahead, returns None.
                """
        next_cactus: Cactus | None = None
        min_distance: float = float('inf')

        for cactus in cacti:
            distance = cactus.x - self.x
            if distance >= 0 and distance < min_distance:
                min_distance = distance
                next_cactus = cactus

        return next_cactus

    def get_second_cactus(self, cacti: list[Cactus]) -> Cactus | None:
        ############################################################
        ###                     AI                               ###
        ############################################################
        """
        Returns the second nearest cactus in front of the player.
        If there is only one or no cactus ahead, returns None.
        """
        ahead_cacti = [cactus for cactus in cacti if cactus.x - self.x >= 0]
        ahead_cacti.sort(key=lambda c: c.x - self.x)

        if len(ahead_cacti) >= 2:
            return ahead_cacti[1]
        elif len(ahead_cacti) == 1:
            return ahead_cacti[0]
        else:
            return None

def check_collisions(players: list[Player], cacti: list[Cactus]) -> list[int]:
    """Return a list of indices of players colliding with any cactus."""
    cactus_rects = [cactus.rect for cactus in cacti]
    colliding_indices = []

    for i, player in enumerate(players):
        collisions = player.rect.collidelistall(cactus_rects)
        if collisions:
            colliding_indices.append(i)

    return colliding_indices


def draw_debug(players: list, outputs: list[list[float]], generation: int, game_speed: float, gnomes: list[neat.DefaultGenome], p: float = p):
    font = pygame.font.SysFont("Arial", 18)

    text_players = font.render(f"Players: {len(players)}", True, (255, 255, 255))
    SCREEN.blit(text_players, (10, 10))

    text_gen = font.render(f"Generation: {generation}", True, (255, 255, 255))
    SCREEN.blit(text_gen, (10, 30))

    text_speed = font.render(f"Game speed: {game_speed:.2f}", True, (255, 255, 255))
    SCREEN.blit(text_speed, (10, 50))

    if outputs:
        avg_outputs = [sum(o[i] for o in outputs) / len(outputs) for i in range(10)]
        avg_text = ", ".join([f"{i}={v:.2f}" for i, v in enumerate(avg_outputs)])
        text_avg = font.render(f"Avg outputs: {avg_text}", True, (255, 255, 0))
        SCREEN.blit(text_avg, (10, 70))

        # Count how many exceeded threshold p
        count_active = [sum(1 for o in outputs if o[i] > p) for i in range(10)]
        count_text = ", ".join([f"{i}={c}" for i, c in enumerate(count_active)])
        text_count = font.render(f"Used outputs: {count_text}", True, (255, 255, 0))
        SCREEN.blit(text_count, (10, 90))

    # Average and max fitness
    if gnomes:
        avg_fitness = sum([g.fitness for g in gnomes]) / len(gnomes)
        max_fitness = max([g.fitness for g in gnomes])
        text_fitness_avg = font.render(f"Avg fitness: {avg_fitness:.2f}", True, (0, 255, 255))
        text_fitness_max = font.render(f"Max fitness: {max_fitness:.2f}", True, (255, 128, 0))
        SCREEN.blit(text_fitness_avg, (10, 110))
        SCREEN.blit(text_fitness_max, (10, 130))

    # NN schematic top-left (further left)
    if outputs:
        for i, out in enumerate(outputs):
            # Convert outputs to -1 / 0 / 1
            discrete_out = [1 if v > p else -1 if v < -p else 0 for v in out[:10]]
            nn_text = font.render(f"NN {i}: {discrete_out}", True, (0, 255, 0))
            SCREEN.blit(nn_text, (SIZE[0] - 350, 10 + i * 20))






def draw(players: list[Player], cacti: list[Cactus]):
    SCREEN.fill((0, 0, 0))
    for player in players:
        player.draw()
    for cactus in cacti:
        cactus.draw()


    pygame.display.flip()
def main(genomes:list[neat.genome], config:neat.config) ->None:
    players: list[Player] = []
    global generation_number, default_game_speed
    generation_number += 1
    if generation_number % 5 == 0:
        default_game_speed += 0.1
    clock = pygame.time.Clock()
    running: bool = True
    cacti: list[Cactus] = [Cactus()]
    cacti[0].set_size()
    nets: list[neat.nn.FeedForwardNetwork] = []
    player_alone_timer: int = 0
    counter: int = 0
    removes: list[Player] = []
    ge: list[neat.genome] = []
    last_jump_level: list[int | None] = []
    repeat_count: list[int] = []
    for x, gnome in genomes: # x, because gnome is a tuple chatgpt says
        gnome.fitness = 0.0
        net: neat.nn.FeedForwardNetwork = neat.nn.FeedForwardNetwork.create(gnome, config)
        nets.append(net)
        players.append(Player())
        ge.append(gnome)
        last_jump_level.append(None)
        repeat_count.append(0)
    gs_timer: int = 0 #Timer for gamespeed
    game_speed: float = default_game_speed
    while running:
        counter += 1
        gs_timer += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    for player in players:
                        player.jump(1)
                if event.key == pygame.K_p:
                    for g in ge:
                        g.fitness += 50
                        running = False
                if event.key == pygame.K_PLUS:
                    game_speed += 0.01

        for i, cactus in enumerate(cacti):
            cactus.update(game_speed)
            if cactus.is_off_screen():
                cacti.pop(i)

        for player in players:
            player.gravity()

        outputs_list = []
        for x, player in enumerate(players):
            # Half AI
            next_cactus = player.get_next_cactus(cacti)
            second_cactus = player.get_second_cactus(cacti)

            output = nets[x].activate((
                player.y,
                player.x,
                player.velocity,
                (next_cactus.x - player.x) if next_cactus else 0,
                next_cactus.width if next_cactus else 0,
                next_cactus.height if next_cactus else 0,
                (second_cactus.x - player.x) if second_cactus else 0,
                second_cactus.width if second_cactus else 0,
                second_cactus.height if second_cactus else 0,
                game_speed,
                player.y - SIZE[1],
                repeat_count[x]
            ))

            for i, o in enumerate(output[:10], start=1):
                if o > p:
                    if player.jump(i):
                        # Keep the same exponential fitness bonus
                        ge[x].fitness += (0.25 * (0.5 ** (i - 1))) * game_speed  # i=1 → 0.25, i=2 → 0.125, etc.

                        # --- NEW: repeated-jump detection and penalty
                        if last_jump_level[x] is not None and last_jump_level[x] == i:
                            repeat_count[x] += 1
                        else:
                            last_jump_level[x] = i
                            repeat_count[x] = 1

                        if repeat_count[x] >= 5:
                            ge[x].fitness -= 0.5
                            repeat_count[x] = 0
                            last_jump_level[x] = None
                    break

            outputs_list.append(output)


        collisions = check_collisions(players, cacti)
        if len(collisions) >= 1:
            for i in sorted(collisions, reverse=True):
                ge[i].fitness-=25
                removes.append(players.pop(i))
                nets.pop(i)
                ge.pop(i)
                last_jump_level.pop(i)
                repeat_count.pop(i)

        if counter >= 50 * math.exp(game_speed / 1000):
            if not cacti or cacti[-1].x < SIZE[0] - random.randint(200, 400):
                cacti.append(Cactus())
                cacti[-1].set_size()
                counter = 0
                for gnome in ge:
                    gnome.fitness += 0.75 * game_speed  # Hilft maybe, wenn sie exponential mehr fitness bekommen


        if gs_timer >= 500*(game_speed-default_game_speed+1)/((generation_number+9)/10):
            game_speed += 0.1
            gs_timer = 0


        if len(players) == 1:
            player_alone_timer += 1
            if player_alone_timer >= 1000:
                ge[0].fitness += 50
                running = False

        draw(players, cacti)
        draw_debug(players, outputs_list, generation_number, game_speed, ge)
        pygame.display.flip()
        clock.tick(1000000)
        file_name = pathlib.Path(f"best_player{generation_number}.svg")
        file_name_2 = pathlib.Path(f"best_player{generation_number}")
        if len(players) == 1:
            if file_name.is_file() or file_name_2.is_file():
                pass
            else:
                visualize.draw_net(config, ge[0], view=True, filename=f"best_player{generation_number}")
        if len(players) == 0:
            running = False



def run(conf):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, conf)
    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    winner = population.run(main)
    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)

    # winner = None
    # with open("winner.pkl", "rb") as f:
    #     winner = pickle.load(f)
    # population = neat.Population(config)
    # if winner:
    #     # Assign an ID not used by NEAT
    # winner = population.run(main)
    #     with open("winner.pkl", "wb") as f:
    #         f.dump(winner)


def run_1_player(conf):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, conf)
    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)


    path = pathlib.Path("winner.pkl")
    winner = None
    with open("winner2.pkl", "rb") as f:
        winner = pickle.load(f)
    if winner:
        new_id = max(population.population.keys()) + 1
        population.population[new_id] = winner

    net = neat.nn.FeedForwardNetwork.create(winner, config)
    winner = population.run(main)

    with open("winner3.pkl", "wb") as f:
        pickle.dump(winner, f)

if __name__ == '__main__':
    #ans = int(input("1 for fresh start \n or 2 for letting an AI play for you."))
    ans = 1
    if ans == 1:
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config_feedforward')
        run(config_path)
    elif ans == 2:
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config_feedforward-1-player')
        run_1_player(config_path)
