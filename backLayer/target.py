import pygame

class Target:
    def __init__(self, position, speed, radius=10, color=(255, 255, 255)):
        super().__init__()
        self.x = position[0]
        self.y = position[1]
        self.speed = speed
        self.radius = radius
        self.color = color
        self.moving = False

    def render(self, screen):
        # Рисуем белый круг
        pygame.draw.circle(
            surface=screen, color=(255, 255, 255), center=(self.x, self.y), radius=self.radius + 1
        )
        pygame.draw.circle(surface=screen, color=self.color, center=(self.x, self.y), radius=self.radius)

    def move(self, target_location, ticks):
        distantion_per_tick = self.speed * ticks / 1000

        # Убеждаемся, что цель проходит фиксированное расстояние в каждом кадре
        if (
            abs(self.x - target_location[0]) <= distantion_per_tick
            and abs(self.y - target_location[1]) <= distantion_per_tick
        ):
            self.moving = False
            # Red color
            self.color = (255, 0, 0)
        else:
            self.moving = True
            # Green color
            self.color = (255, 255, 0)
            current_vector = pygame.Vector2(x=self.x, y=self.y)
            new_vector = pygame.Vector2(x= target_location[0], y= target_location[1])
            # Направление
            towards = (new_vector - current_vector).normalize()

            self.x += towards[0] * distantion_per_tick
            self.y += towards[1] * distantion_per_tick