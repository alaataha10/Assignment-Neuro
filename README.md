import numpy as np
vocab = ['I', 'love', 'Dr', 'Noha']
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}

inputs = [word_to_idx['I'], word_to_idx['love'], word_to_idx['Dr']] 
target = word_to_idx['Noha']  

vocab_size = len(vocab)
hidden_size = 8 
learning_rate = 0.01

Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  
Why = np.random.randn(vocab_size, hidden_size) * 0.01  
bh = np.zeros((hidden_size, 1))  
by = np.zeros((vocab_size, 1))   

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def train_step(inputs, target, Wxh, Whh, Why, bh, by):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.zeros((hidden_size, 1))

    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))  
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
    
    ys = np.dot(Why, hs[len(inputs)-1]) + by
    ps = softmax(ys)

    loss = -np.log(ps[target, 0])

    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])

    dy = np.copy(ps)
    dy[target] -= 1

    dWhy += np.dot(dy, hs[len(inputs)-1].T)
    dby += dy

    dh = np.dot(Why.T, dy) + dhnext
    for t in reversed(range(len(inputs))):
        dhraw = (1 - hs[t] * hs[t]) * dh  
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dh = np.dot(Whh.T, dhraw)

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)

    Wxh -= learning_rate * dWxh
    Whh -= learning_rate * dWhh
    Why -= learning_rate * dWhy
    bh -= learning_rate * dbh
    by -= learning_rate * dby

    return loss, ps, Wxh, Whh, Why, bh, by

for epoch in range(300):
    loss, probs, Wxh, Whh, Why, bh, by = train_step(inputs, target, Wxh, Whh, Why, bh, by)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

def predict(inputs, Wxh, Whh, Why, bh, by):
    hs = np.zeros((hidden_size, 1))
    for t in range(len(inputs)):
        x = np.zeros((vocab_size, 1))
        x[inputs[t]] = 1
        hs = np.tanh(np.dot(Wxh, x) + np.dot(Whh, hs) + bh)
    y = np.dot(Why, hs) + by
    p = softmax(y)
    idx = np.argmax(p)
    return idx_to_word[idx]

predicted_word = predict(inputs, Wxh, Whh, Why, bh, by)
print(f"Predicted word: {predicted_word}")
