# number_of_weights = 6

current_Q = np.zeros((number_of_states, 2))

# W = rand(1, number_of_weights)
# W(1, number_of_weights) = 1

for i in range(number_of_states):
    j = i
    while True:
        state = states_hash[j]
        new_state = state

        action
        if random() <= 1 - epsilon:
            if current_Q[j, 1] >= current_Q[j, 2]:
                action = 1
            else:
                action = 2
        else:
            if random() <= 0.5:
                action = 1
            else:
                action = 2
            
        state_reward = 0

        new_arrival_a = false
        new_arrival_b = false

        if random() <= p_a:
            new_arrival_a = True

        if random() <= p_b:
            new_arrival_b = True

        working_queue = state[3]
        other_queue = 1
        if working_queue == 1:
            other_queue = 2
        elif working_queue == 2:
            other_queue = 1

        if action == 1:
            if new_arrival_a and state[1] != buffer_size:
                new_state[1] = new_state[1] + 1

            if new_arrival_b and new_state[2] != buffer_size:
                new_state[2] = new_state[2] + 1

            new_state[3] = other_queue

        else:
            if state(working_queue) != 0:
                state_reward = reward

            arrivals = [new_arrival_a, new_arrival_b]

            if arrivals[other_queue] and state[other_queue] != buffer_size:
                new_state[other_queue] = new_state[other_queue] + 1

            if arrivals[working_queue]:
                if state[working_queue] == buffer_size:
                    new_state[working_queue] = new_state[working_queue] - 1
            else:
                if state[working_queue] != 0:
                    new_state[working_queue] = new_state[working_queue] - 1

    k = 0
    while k < number_of_states:
        if states_hash[k] == new_state:
            break
        k = k + 1

    action1
    if current_Q[k, 1] >= current_Q[k, 2]:
        action1 = 1
    else:
        action1 = 2

    # X_j = [states_hash[j, 1], states_hash[j, 2], action, states_hash[j, 1] * action, states_hash[j, 2] * action, 0]

    # W = W + alpha * (state_reward + gamma * current_Q(k, action1) - current_Q(j, action1)) * X_j

    # for m = 1: number_of_states
        
    #     new_approximation1 = W(1) * states_hash{m}(1) + W(2) * states_hash{m}(2) + W(3) * 1 + 1 * W(4) * states_hash{m}(1) + W(5) * 1 * states_hash{m}(2) + W(6)
    #     new_approximation2 = W(1) * states_hash{m}(1) + W(2) * states_hash{m}(2) + W(3) * 2 + 2 * W(4) * states_hash{m}(1) + W(5) * 2 * states_hash{m}(2) + W(6)

    #     current_Q(m, 1) = new_approximation1
    #     current_Q(m, 2) = new_approximation2

    new_actions = np.zeros((number_of_states, 1))
    for n in range(number_of_states):
        if current_Q[n, 1] >= current_Q[n, 2]:
            new_actions[n, 1] = 1
        else:
            new_actions[n, 1] = 2

    if policy == new_actions:
        break

    j = k
