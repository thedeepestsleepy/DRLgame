import pygame
from environment import RedSquareEnv

def main():

    pygame.init()
    clock = pygame.time.Clock()


    env = RedSquareEnv()
    env.reset()
    done = False


    while not done:
 
        action = 0  
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                env.close()
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 1 
                elif event.key == pygame.K_RIGHT:
                    action = 2 
                elif event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                    action = 3

    
        obs, reward, done, info = env.step(action)

 
        env.render(mode='background')

   
        clock.tick(60)


        if done:

            font = pygame.font.SysFont('Arial', 30)
            if 'game_over' in info:
                if info['game_over'] == 'win':
                    message = "Win"
                elif info['game_over'] == 'lose':
                    message = "Game Over"
            else:
                message = "Game Over"

            text_surface = font.render(message, True, (255, 255, 255))
            env.display_screen.blit(
                text_surface,
                ((env.WIDTH - text_surface.get_width()) // 2, (env.HEIGHT - text_surface.get_height()) // 2)
            )
            pygame.display.flip()

            pygame.time.wait(3000)
            done = True


    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()