import numpy as np
import keras
from keras import layers
from random import random, randint, sample

# Q-learning with function approximation

aa = np.array([[1, 2, 3], [4, 5, 6]])

print(aa[:, -1].shape)

exit()

p_a = 0.1
p_b = 0.7

gamma = 0.99
alpha = 0.1
epsilon = 0.05

buffer_size = 30

number_of_queues = 2
number_of_acitons = 2
number_of_states = (1 + buffer_size) * (1 + buffer_size) * number_of_queues

reward = 1

states_hash = []

for i in range(1, number_of_queues + 1):
    for j in range(buffer_size + 1):
        for k in range(buffer_size + 1):
            states_hash.append([j, k, i])


states_hash = np.array(states_hash)

replay_memory_capacity = 500
replay_memory = np.array([[0, 0, 0, 0]])
replay_memory = np.delete(replay_memory, (0), axis=0)

policy_model = keras.Sequential(
    [
        layers.Dense(4, activation="relu", name="layer1"),
        layers.Dense(16, activation="relu", name="layer2"),
        layers.Dense(8, activation="relu", name="layer3"),
        layers.Dense(2, name="layer4"),
    ]
)

policy_model.build(input_shape=(1, 4))
policy_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error')


target_model = keras.models.clone_model(policy_model)
target_model.build(input_shape=(1, 4))
target_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error')
target_model.set_weights(policy_model.get_weights())

current_Q = np.zeros((number_of_states, number_of_acitons))

for e in range(100000):
    starting_state = randint(0, number_of_states - 1)
    j = starting_state

    for t in range(100000):

        curr_state = np.copy(states_hash[j])
        new_state = np.copy(states_hash[j])

        action = None
        if random() <= 1 - epsilon:
            if current_Q[j, 0] >= current_Q[j, 1]:
                action = 1
            else:
                action = 2
        else:
            if random() <= 0.5:
                action = 1
            else:
                action = 2
            
        state_reward = 0

        new_arrival_a = False
        new_arrival_b = False

        if random() <= p_a:
            new_arrival_a = True

        if random() <= p_b:
            new_arrival_b = True

        working_queue = curr_state[2]
        other_queue = 1
        if working_queue == 1:
            other_queue = 2
        elif working_queue == 2:
            other_queue = 1

        if action == 1:
            if new_arrival_a and curr_state[0] != buffer_size:
                new_state[0] = new_state[0] + 1

            if new_arrival_b and new_state[1] != buffer_size:
                new_state[1] = new_state[1] + 1

            new_state[2] = other_queue
        else:
            if curr_state[working_queue - 1] != 0:
                state_reward = reward

            arrivals = [new_arrival_a, new_arrival_b]

            if arrivals[other_queue - 1] and curr_state[other_queue - 1] != buffer_size:
                new_state[other_queue - 1] = new_state[other_queue - 1] + 1

            if arrivals[working_queue - 1]:
                if curr_state[working_queue - 1] == buffer_size:
                    new_state[working_queue - 1] = new_state[working_queue - 1] - 1
            else:
                if curr_state[working_queue - 1] != 0:
                    new_state[working_queue - 1] = new_state[working_queue - 1] - 1

        k = 0
        while k < number_of_states:
            if np.all(np.equal(states_hash[k], new_state)):
                break
            k = k + 1

        if len(replay_memory) < replay_memory_capacity:
                replay_memory = np.vstack([replay_memory, np.array([j, action, k, state_reward])])
        else:
            replay_memory = np.delete(replay_memory, (0), axis=0)
            replay_memory = np.vstack([replay_memory, np.array([j, action, k, state_reward])])

        policy_network_input = np.array([[0, 0, 0, 0]])
        policy_network_input = np.delete(policy_network_input, (0), axis=0)

        target_network_input = np.array([[0, 0, 0, 0]])
        target_network_input = np.delete(target_network_input, (0), axis=0)

        rnumber = None
        if len(replay_memory) >= 16:
            rnumber = sample(range(0, len(replay_memory)), 16)
            
            for l in rnumber:
                s_1 = replay_memory[l, 0]
                #policy_network_input = np.vstack([policy_network_input, np.array([j, states_hash[s_1, 0], states_hash[s_1, 1], states_hash[s_1, 2], replay_memory[l, 1], k, states_hash[s_2, 0], states_hash[s_2, 1], states_hash[s_2, 2], replay_memory[l, 3]])])
                policy_network_input = np.vstack([policy_network_input, np.array([replay_memory[l, 0], states_hash[s_1, 0], states_hash[s_1, 1], states_hash[s_1, 2]])])
            
        if len(policy_network_input) > 0:
            # policy_network_input = policy_network_input.reshape((1, 160))
            print(policy_network_input)
            print(policy_network_input.shape)
            # policy_model.train_on_batch(x=policy_network_input, y=None, sample_weight=None, class_weight=None, return_dict=False)
            ans_1 = policy_model.predict(x=policy_network_input, batch_size=None, verbose="auto", steps=None, callbacks=None)

            for l in range(len(ans_1)):
                current_Q[policy_network_input[l, 0]] = np.copy(ans_1[l])

            for l in len(ans_1):
                new_ans_1 = np.array([[0]])
                new_ans_1 = np.delete(new_ans_1, (0), axis=0)
                new_ans_1 = np.vstack([new_ans_1, np.array([ans_1[l, replay_memory[rnumber[l], 1] - 1]])])
            

            for l in rnumber:
                s_2 = replay_memory[l, 2]

                #target_network_input = np.vstack([target_network_input, np.array([j, states_hash[s_1, 0], states_hash[s_1, 1], states_hash[s_1, 2], replay_memory[l, 1], k, states_hash[s_2, 0], states_hash[s_2, 1], states_hash[s_2, 2], replay_memory[l, 3]])])
                target_network_input = np.vstack([target_network_input, np.array([replay_memory[l, 2], states_hash[s_2, 0], states_hash[s_2, 1], states_hash[s_2, 2]])])


            ans_2 = target_model.predict(x=target_network_input, batch_size=None, verbose="auto", steps=None, callbacks=None)

            new_ans_2 = np.max(ans_2, axis=1)

            state_rewards = np.array([[0]])
            state_rewards = np.delete(state_rewards, (0), axis=0)
            for l in rnumber:
                state_rewards = np.vstack([state_rewards, np.array([replay_memory[l, -1]])])
            

                


            
            
            
            # print(ans)
            exit()

        



        action1 = None
        if current_Q[k, 0] >= current_Q[k, 1]:
            action1 = 1
        else:
            action1 = 2

        j = k

