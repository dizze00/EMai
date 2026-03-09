def respond(fused):
    emotion = fused["emotion"]

    if emotion == "angry":
        return "Calm down, Anger is high!"

    return "I'm listening."