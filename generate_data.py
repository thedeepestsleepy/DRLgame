from environment import RedSquareEnv

def generate_data():
    env = RedSquareEnv(data_generation=True)
    env.reset()  
    for i in range(1000):  

        env.step(0)
        print('idx',i)
    env.close()

if __name__ == "__main__":
    generate_data()