import numpy as np
from hmmlearn import hmm

states = ["e", "v"]
observations = ["High", "Low"]

def create_word_model(start_prob, trans_mat, emiss_mat):

    model = hmm.CategoricalHMM(n_components=2)
    model.startprob_ = start_prob
    model.transmat_ = trans_mat
    model.emissionprob_ = emiss_mat
    return model

# P(e)=1.0, P(v)=0.0
start_ev = np.array([1.0, 0.0])

# P(e->e)=0.6, P(e->v)=0.4
# P(v->v)=0.8, P(v->e)=0.2
trans_ev = np.array([
    [0.6, 0.4],
    [0.2, 0.8]
])

#(B):
# e durumunda: High=0.7, Low=0.3
# v durumunda: High=0.1, Low=0.9
emiss_ev = np.array([
    [0.7, 0.3],
    [0.1, 0.9]
])

model_ev = create_word_model(start_ev, trans_ev, emiss_ev)

start_okul = np.array([0.5, 0.5])
trans_okul = np.array([[0.5, 0.5], [0.5, 0.5]])
emiss_okul = np.array([[0.3, 0.7], [0.7, 0.3]])

model_okul = create_word_model(start_okul, trans_okul, emiss_okul)

def classify_speech(obs_sequence):
    obs_array = np.array([obs_sequence]).T
    score_ev = model_ev.score(obs_array)
    score_okul = model_okul.score(obs_array)

    print("-" * 30)
    print(f"Gözlem Dizisi: {[observations[i] for i in obs_sequence]}")
    print(f"EV Modeli Log-Likelihood: {score_ev:.4f}")
    print(f"OKUL Modeli Log-Likelihood: {score_okul:.4f}")

    if score_ev > score_okul:
        print("Sonuç: Bu ses kaydı 'EV' kelimesine benziyor.")
    else:
        print("Sonuç: Bu ses kaydı 'OKUL' kelimesine benziyor.")
    print("-" * 30)

def find_best_path(obs_sequence):
    obs_array = np.array([obs_sequence]).T
    logprob, state_sequence = model_ev.decode(obs_array, algorithm="viterbi")
    path = [states[i] for i in state_sequence]
    return path

if __name__ == "__main__":
    test_obs = [0,1]
    classify_speech(test_obs)
    best_path = find_best_path(test_obs)
    print(f"Viterbi Algoritmasına Göre En Olası Fonem Dizisi: {best_path}")