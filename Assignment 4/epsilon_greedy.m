function a = epsilon_greedy(Q, state, epsilon)
    r = rand;
    if r <= epsilon
        a = randsample(Q(state,:));
    else
        [~, a] = max(Q(state,:));
    end
end