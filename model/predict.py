import util
import pickle
import io
import pandas as pd
import base64
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("agg")


def MBTI(input):
    tokenize = util.build_bag_of_words_features_filtered(input)
    # Load Introvert-Extrovert model
    with open("intro_extro_model.pkl", "rb") as file:
        IntroExtro = pickle.load(file)

    # Load Intuition-Sensing model
    with open("intuition_sensing_model.pkl", "rb") as file:
        IntuitionSensing = pickle.load(file)

    # Load Thinking-Feeling model
    with open("thinking_feeling_model.pkl", "rb") as file:
        ThinkingFeeling = pickle.load(file)

    # Load Judging-Perceiving model
    with open("judging_perceiving_model.pkl", "rb") as file:
        JudgingPercieiving = pickle.load(file)

    ie = IntroExtro.classify(tokenize)
    Is = IntuitionSensing.classify(tokenize)
    tf = ThinkingFeeling.classify(tokenize)
    jp = JudgingPercieiving.classify(tokenize)

    mbt = ""

    if ie == "introvert":
        mbt += "I"
    if ie == "extrovert":
        mbt += "E"
    if Is == "Intuition":
        mbt += "N"
    if Is == "Sensing":
        mbt += "S"
    if tf == "Thinking":
        mbt += "T"
    if tf == "Feeling":
        mbt += "F"
    if jp == "Judging":
        mbt += "J"
    if jp == "Percieving":
        mbt += "P"
    return mbt


def tellmemyMBTI(input, name, traasits=[]):
    print("predicting")
    a = []
    trait1 = pd.DataFrame([0, 0, 0, 0], ["I", "N", "T", "J"], ["count"])
    trait2 = pd.DataFrame([0, 0, 0, 0], ["E", "S", "F", "P"], ["count"])
    for i in input:
        a += [MBTI(i)]
    for i in a:
        for j in ["I", "N", "T", "J"]:
            if j in i:
                trait1.loc[j] += 1
        for j in ["E", "S", "F", "P"]:
            if j in i:
                trait2.loc[j] += 1
    trait1 = trait1.T
    trait1 = trait1 * 100 / len(input)
    trait2 = trait2.T
    trait2 = trait2 * 100 / len(input)

    # Finding the personality
    YourTrait = ""
    for i, j in zip(trait1, trait2):
        temp = max(trait1[i][0], trait2[j][0])
        if trait1[i][0] == temp:
            YourTrait += i
        elif trait2[j][0] == temp:
            YourTrait += j
    traasits += [YourTrait]

    # Plotting
    temp = {
        "train": [0] * 4,
        "test": [0] * 4,
    }

    results = pd.DataFrame.from_dict(
        temp,
        orient="index",
        columns=[
            "Introvert - Extrovert",
            "Intuition - Sensing",
            "Thinking - Feeling",
            "Judging - Percieiving",
        ],
    )
    labels = np.array(results.columns)

    intj = trait1.loc["count"]
    ind = np.arange(4)
    width = 0.4
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    rects1 = ax.bar(ind, intj, width, color="royalblue")

    esfp = trait2.loc["count"]
    rects2 = ax.bar(ind + width, esfp, width, color="seagreen")
    fig.set_size_inches(10, 7)
    plt.ylim(intj.max())
    ax.set_xlabel("Finding the MBTI Trait")
    ax.set_ylabel("Trait Percent (%)")
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(0, 100, step=10))
    ax.set_title("Your Personality is " + YourTrait, size=20)
    plt.grid(True)
    my_stringIObytes = io.BytesIO()
    plt.savefig(my_stringIObytes, format="jpg")
    my_stringIObytes.seek(0)
    base64_data = base64.b64encode(my_stringIObytes.read()).decode()
    return (YourTrait, base64_data)
