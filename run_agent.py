from env import Environment
from agent_brain import QLearningTable


def update():
    steps = []
    all_costs = []
    epoch=1000

    for episode in range(epoch):
        observation = env.reset()
        i = 0
        cost = 0

        while True:
            env.render()
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            cost += RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_
            i += 1

            if done:
                steps += [i]
                all_costs += [cost]
                break

    env.final()

    RL.print_q_table()

    RL.plot_results(steps, all_costs)


if __name__ == "__main__":
    # Calling for the environment
    env = Environment()
    # Calling for the main algorithm
    RL = QLearningTable(actions=list(range(env.n_actions)))
    # Running the main loop with Episodes by calling the function update()
    env.after(0, update)  # Or just update()
    env.mainloop()
