from tqdm import tqdm


def compose_samples(samples, task, sep_token=None):
    composed_samples = COMPOSITION[task](samples, sep_token=sep_token)
    return composed_samples


def compose_text_topic_samples(samples, sep_token):
    if sep_token is not None:
        composed_samples = samples.apply(
            lambda row: row["text"] + " " + sep_token + " " + row["topic"]
            , axis=1)
    else:
        composed_samples = samples.apply(
            lambda row: row["text"] + " " + row["topic"]
            , axis=1)

    return composed_samples


def compose_text_samples(samples, sep_token):
    composed_samples = samples.apply(
        lambda row: row["text"]
        , axis=1)
    return composed_samples


COMPOSITION = {
    "x-stance-de": compose_text_topic_samples,
    "germeval-toxic": compose_text_samples,
    "germeval-factclaiming": compose_text_samples,
    "germeval-engaging": compose_text_samples,
    "detox-toxic": compose_text_samples,
    "detox-hate-speech": compose_text_samples,
    "detox-sentiment": compose_text_samples,
}

LABEL_MAPPING = {
    "x-stance-de": [{
        0: "dagegen",
        1: "dafür",
    }, {
        0: "dagegen",
        1: "dafür",
    }, {
        0: "dagegen",
        1: "dafür",
    }, {
        0: "dagegen",
        1: "dafür",
    }, {
        0: "favor",
        1: "against",
    }, {
        0: "favor",
        1: "against",
    }, {
        0: "favor",
        1: "against",
    }, {
        0: "favor",
        1: "against",
    }],
    "germeval-toxic": [{
        0: "nein",
        1: "ja",
    }, {
        0: "nein",
        1: "ja",
    }, {
        0: "nein",
        1: "ja",
    }, {
        0: "nein",
        1: "ja",
    }, {
        0: "no",
        1: "yes",
    }, {
        0: "no",
        1: "yes",
    }, {
        0: "no",
        1: "yes",
    }, {
        0: "no",
        1: "yes",
    }],
    "germeval-factclaiming": [{
        0: "nein",
        1: "ja",
    }, {
        0: "nein",
        1: "ja",
    }, {
        0: "nein",
        1: "ja",
    }, {
        0: "nein",
        1: "ja",
    }, {
        0: "no",
        1: "yes",
    }, {
        0: "no",
        1: "yes",
    }, {
        0: "no",
        1: "yes",
    }, {
        0: "no",
        1: "yes",
    }],
    "detox-sentiment": [{
        0: "neutral",
        1: "negativ",
        2: "positiv",
    }, {
        0: "neutral",
        1: "negativ",
        2: "positiv",
    }, {
        0: "neutral",
        1: "negativ",
        2: "positiv",
    }, {
        0: "neutral",
        1: "negativ",
        2: "positiv",
    }],
    "detox-hate-speech": [{
        0: "nein",
        1: "ja",
    }, {
        0: "nein",
        1: "ja",
    }, {
        0: "nein",
        1: "ja",
    }, {
        0: "nein",
        1: "ja",
    }],
    "detox-toxic": [{
        0: "nein",
        1: "ja",
    }, {
        0: "nein",
        1: "ja",
    }, {
        0: "nein",
        1: "ja",
    }, {
        0: "nein",
        1: "ja",
    }],
    "germeval-engaging": [{
        0: "nein",
        1: "ja",
    }, {
        0: "nein",
        1: "ja",
    }, {
        0: "nein",
        1: "ja",
    }, {
        0: "nein",
        1: "ja",
    }, {
        0: "no",
        1: "yes",
    }, {
        0: "no",
        1: "yes",
    }, {
        0: "no",
        1: "yes",
    }, {
        0: "no",
        1: "yes",
    }],
}


def x_stance_de_instruction(example_label):
    instruction = "Was ist die Haltung des folgenden Satzes gegenüber dem Thema?"

    if example_label:
        instruction += " Antworte mit einem Wort, entweder mit 'dafür' falls der Satz das Thema unterstützt oder 'dagegen' falls der Satz das Thema nicht unterstützt."

    instruction += "\n"
    return instruction


def germeval_toxic_instruction(example_label):
    instruction = "Ist der folgende Satz toxisch?"

    if example_label:
        instruction += " Antworte mit einem Wort, entweder 'ja' falls der Satz toxisch ist oder 'nein' falls nicht."

    instruction += "\n"
    return instruction


def germeval_toxic_chat_instruction(example_label):
    instruction = (
        "Geben ist der folgenden Satz, ist dieser toxisch oder nicht? "
        "Mögliche Antworten sind 'ja', falls der Satz toxisch ist oder 'nein' falls nicht. "
        "Antworte nur mit einem dieser Möglichkeiten und ohne Erklärung!"
    )

    return instruction


def germeval_toxic_chat_instruction_1(example_label):
    instruction = (
        "Die Aufgabe ist es zu erkennen, ob der folgende Satz toxisch ist oder nicht. "
        "Mögliche Antworten sind 'ja', falls der Satz toxisch ist oder 'nein' falls nicht. "
        "Antworte nur mit einem dieser Möglichkeiten und ohne Erklärung!"
    )

    return instruction


def germeval_toxic_chat_instruction_2(example_label):
    instruction = (
        "Betrachten wir den folgenden Satz. Ist dieser toxisch? "
        "Antworte mit einem Wort: 'ja', falls der Satz toxisch ist oder 'nein' falls nicht."
    )

    return instruction


def germeval_toxic_chat_instruction_5(example_label):
    instruction = "Geben ist der folgenden Satz, ist dieser toxisch oder nicht?"

    if example_label:
        instruction += " Answer with just one word and no explanation! Either 'yes' if the sentence is toxic or 'no' if not."

    return instruction


def germeval_factclaiming_chat_instruction(example_label):
    instruction = (
        "Geben ist der folgenden Satz, wird in diesem Tatsachen behauptet oder nicht? "
        "Mögliche Antworten sind 'ja', falls im Satz Tatsachen behauptet werden oder ‘nein' falls nicht. "
        "Antworte nur mit einem dieser Möglichkeiten und ohne Erklärung!"
    )

    return instruction


def germeval_factclaiming_chat_instruction_1(example_label):
    instruction = (
        "Die Aufgabe ist es zu erkennen ob im folgenden Satz Tatsachen behauptet werden oder nicht. "
        "Mögliche Antworten sind 'ja', falls im Satz Tatsachen behauptet werden oder ‘nein' falls nicht. "
        "Antworte nur mit einem dieser Möglichkeiten und ohne Erklärung!"
    )

    return instruction


def germeval_factclaiming_chat_instruction_2(example_label):
    instruction = (
        "Betrachten wir den folgenden Satz. Wird in diesem Tatsachen behauptet? "
        "Antworte mit einem Wort: 'ja' falls der Satz Tatsachen behauptet oder 'nein' falls nicht."
    )

    return instruction


def germeval_factclaiming_instruction(example_label):
    instruction = "Deine Aufgabe ist es zu erkennen ob im folgende Satz Tatsachen erwähnt werden oder nicht."

    if example_label:
        instruction += " Antworte mit einem Wort, entweder 'ja' falls der Satz Tatsachen erwähnt oder 'nein' falls nicht."

    instruction += "\n"
    return instruction


def germeval_engaging_instruction(example_label):
    instruction = "Deine Aufgabe ist es zu erkennen ob der folgende Satz fesselnd ist oder nicht."

    if example_label:
        instruction += " Antworte mit einem Wort, entweder 'ja' falls der Satz fesselnd ist oder 'nein' falls nicht."

    instruction += "\n"
    return instruction


def detox_hatespeech_chat_instruction(example_label):
    instruction = (
        "Geben ist der folgenden Satz, enthält dieser Hassrede? "
        "Mögliche Antworten sind 'ja', falls der Satz Hassrede enthält oder ‘nein' falls nicht. "
        "Antworte nur mit einem Wort und ohne Erklärung!"
    )
    return instruction


def detox_hatespeech_chat_instruction_1(example_label):
    instruction = (
        "Die Aufgabe ist es zu erkennen ob der folgende Satz Hassrede enthält. "
        "Mögliche Antworten sind 'ja', falls der Satz Hassrede enthält oder ‘nein' falls nicht. "
        "Antworte nur mit einem Wort und ohne Erklärung!"
    )
    return instruction


def detox_hatespeech_chat_instruction_2(example_label):
    instruction = (
        "Betrachten wir den folgenden Satz. Enthält dieser Hassrede? "
        "Antworte mit einem Wort: 'ja' falls der Satz Hassrede enthält oder ‘nein' falls nicht. "
    )

    return instruction


def germeval_engaging_chat_instruction(example_label):
    instruction = (
        "Geben ist der folgenden Satz, ist dieser fesselnd oder nicht? "
        "Mögliche Antworten sind 'ja', falls der Satz fesselnd ist oder ‘nein' falls nicht. "
        "Antworte nur mit einem Wort und ohne Erklärung!"
    )
    return instruction


def germeval_engaging_chat_instruction_1(example_label):
    instruction = (
        "Die Aufgabe ist es zu erkennen ob der folgende Satz fesselnd ist oder nicht. "
        "Mögliche Antworten sind 'ja', falls der Satz fesselnd ist oder ‘nein' falls nicht. "
        "Antworte nur mit einem Wort und ohne Erklärung!"
    )
    return instruction


def germeval_engaging_chat_instruction_2(example_label):
    instruction = (
        "Betrachten wir den folgenden Satz. Ist dieser fesselnd? "
        "Antworte mit einem Wort: 'ja' falls der Satz fesselnd ist oder 'nein' falls nicht."
    )

    return instruction


def detox_sentiment_chat_instruction(example_label):
    instruction = (
        "Geben ist der folgenden Satz, was ist der Sentiment? "
        "Mögliche Antworten sind 'negative', 'neutral' oder 'positiv'. "
        "Antworte nur mit einem Wort und ohne Erklärung!"
    )
    return instruction


def detox_sentiment_chat_instruction_1(example_label):
    instruction = (
        "Die Aufgabe ist es den Sentiment des folgenden Satzes zu erkennen."
        "Mögliche Antworten sind 'negative', 'neutral' oder 'positiv'. "
        "Antworte nur mit einem Wort und ohne Erklärung!"
    )
    return instruction


def detox_sentiment_chat_instruction_2(example_label):
    instruction = (
        "Betrachten wir den folgenden Satz. Was ist sein Sentiment? "
        "Antworte mit einem Wort: 'negative', 'neutral' oder 'positiv'."
    )

    return instruction


def compose_x_stance_de_demonstration_query_1(row):
    instruction = "Satz: " + row["text"] + "\n"
    instruction += "Thema: " + row["topic"] + "\n"
    instruction += "Haltung:"

    return instruction


def compose_germeval_toxic_demonstration_query_1(row):
    instruction = "Satz: " + row["text"] + "\n"
    instruction += "Toxisch:"

    return instruction


def compose_detox_sentiment_demonstration_query_1(row):
    instruction = "Satz: " + row["text"] + "\n"
    instruction += "Sentiment:"

    return instruction


def compose_detox_hatespeech_demonstration_query_1(row):
    instruction = "Satz: " + row["text"] + "\n"
    instruction += "Hassrede:"

    return instruction


def compose_germeval_factclaiming_demonstration_query_1(row):
    instruction = "Satz: " + row["text"] + "\n"
    instruction += "Tatsachen erwähnt:"

    return instruction


def compose_germeval_engaging_demonstration_query_1(row):
    instruction = "Satz: " + row["text"] + "\n"
    instruction += "Fesselnd:"

    return instruction


def compose_x_stance_de_chat_demonstration_query_1(row):
    instruction = f"Satz: " + row["text"] + "\n"
    instruction += "Thema: " + row["topic"] + "\n"
    instruction += "Antwort:"

    return instruction


def compose_x_stance_de_chat_demonstration_query_2(row):
    instruction = f'Was ist die Haltung von dem Satz "{row["text"]}" gegenüber dem Thema "{row["topic"]}"?' \
                  f' Antworte mit "dafür", falls der Satz das Thema unterstützt oder "dagegen" falls der Satz das Thema nicht unterstützt.'

    return instruction


def compose_germeval_toxic_chat_demonstration_query_1(row):
    instruction = "Satz: " + row["text"] + "\n"
    instruction += "Antwort:"

    return instruction


def compose_germeval_toxic_chat_demonstration_query_2(row):
    instruction = f'Ist der Satz "{row["text"]}" toxisch oder nicht?' \
                  f' Antworte mit "ja", falls der Satz toxisch ist oder "nein" falls nicht.'

    return instruction


def compose_germeval_factclaiming_chat_demonstration_query_2(row):
    instruction = f'Werden im Satz "{row["text"]}" Tatsachen behauptet oder nicht' \
                  f' Antworte mit "ja", falls der Satz toxisch ist oder "nein" falls nicht.'

    return instruction


def compose_germeval_factclaiming_chat_demonstration_query_1(row):
    instruction = "Satz: " + row["text"] + "\n"
    instruction += "Antwort:"

    return instruction


def compose_germeval_engaging_chat_demonstration_query_1(row):
    instruction = "Satz: " + row["text"] + "\n"
    instruction += "Antwort:"

    return instruction


def compose_germeval_engaging_chat_demonstration_query_2(row):
    instruction = f'Ist der Satz "{row["text"]}" fesselnd oder nicht?' \
                  f' Antworte mit "ja", falls der Satz fesselnd ist oder "nein" falls nicht.'

    return instruction


def compose_detox_hatespeech_chat_demonstration_query_1(row):
    instruction = "Satz: " + row["text"] + "\n"
    instruction += "Antwort:"

    return instruction


def compose_detox_hatespeech_chat_demonstration_query_2(row):
    instruction = f'Enthält der Satz "{row["text"]}" Hassrede oder nicht?' \
                  f' Antworte mit "ja", falls der Satz Hassrede enthält oder "nein" falls nicht.'

    return instruction


def compose_detox_sentiment_chat_demonstration_query_1(row):
    instruction = "Satz: " + row["text"] + "\n"
    instruction += "Antwort:"

    return instruction


def compose_detox_sentiment_chat_demonstration_query_2(row):
    instruction = f'Was ist der Sentiment des Satzes "{row["text"]}"?' \
                  f' Antworte entweder mit "negativ", "neutral" oder "positiv".'

    return instruction


def empty_chat_instruction(example_label):
    return ""


def x_stance_de_chat_instruction(example_label):
    instruction = (
        "Geben dem folgenden Satz und Thema, was ist die Haltung des Satzes gegenüber dem Thema? "
        "Entweder ‘dafür’, falls der Satz das Thema unterstützt oder 'dagegen' falls der Satz das Thema nicht unterstützt. "
        "Antworte nur mit einem dieser Möglichkeiten und ohne Erklärung!"
    )

    return instruction


def x_stance_de_chat_instruction_1(example_label):
    instruction = (
        "Die Aufgabe ist es die Haltung von einem Satz gegenüber dem gegebenen Thema zu erkennen. "
        "Entweder ‘dafür’, falls der Satz das Thema unterstützt oder 'dagegen' falls der Satz das Thema nicht unterstützt. "
        "Antworte nur mit einem dieser Möglichkeiten und ohne Erklärung!"
    )
    return instruction


def x_stance_de_chat_instruction_2(example_label):
    instruction = (
        "Betrachten wir den folgenden Satz und Thema. Unterstützt der Satz das gegeben Thema oder nicht? "
        "Antworte dabei nur mit einem Wort: ‘dafür’, falls der Satz das Thema unterstützt oder 'dagegen' falls der Satz das Thema nicht unterstützt."
    )

    return instruction


def compose_chat_instructions(example_samples, samples, task, seed, k, template_index=0, example_label=True,
                              model_name="", scores=None, prefix_instruction=None):
    sample_instructions = []

    random_state = seed

    for i, row in tqdm(samples.iterrows(), desc='compose instructions', ):

        instructions = []

        if "text" in row:
            other_samples = example_samples[example_samples["text"] != row["text"]]
        elif "premise" in row:
            other_samples = example_samples[
                (example_samples["hypothesis"] != row["hypothesis"]) & (example_samples["premise"] != row["premise"])
                ]
        else:
            other_samples = example_samples[
                (example_samples["text_1"] != row["text_1"]) & (example_samples["text_2"] != row["text_2"])
                ]

        if scores:
            top_indices = scores[i][-k * 4:][::-1]
            top_indices = [top_index for top_index in top_indices if top_index in other_samples.index]

            try:
                instruction_example_samples = other_samples.loc[top_indices[:k]]
            except:
                print()
        else:
            instruction_example_samples = other_samples.sample(k, random_state=random_state)

        random_state = random_state + 1

        instruction_example_samples["label"] = instruction_example_samples["label"]

        instruction = CHAT_INSTRUCTION[task][template_index](example_label)

        if prefix_instruction != None:
            instruction = f"{prefix_instruction}\n{instruction}"

        if instruction != "":
            instructions.append({
                "role": "system",
                "content": instruction
            })

        if k > 0:
            for i, demonstration_row in instruction_example_samples.iterrows():
                instructions.append({
                    "role": "user",
                    "content": CHAT_DEMONSTRATION_QUERY[task][template_index](demonstration_row)
                })
                instructions.append({
                    "role": "assistant",
                    "content": LABEL_MAPPING[task][template_index][demonstration_row["label"]]
                })

        instructions.append({
            "role": "user",
            "content": CHAT_DEMONSTRATION_QUERY[task][template_index](row)
        })

        sample_instructions.append({
            "label": row["label"],
            "instructions": instructions
        })

    return sample_instructions


CHAT_INSTRUCTION = {
    "x-stance-de": [
        x_stance_de_chat_instruction, x_stance_de_chat_instruction_1, x_stance_de_chat_instruction_2,
        empty_chat_instruction,
    ],
    "germeval-toxic": [
        germeval_toxic_chat_instruction, germeval_toxic_chat_instruction_1, germeval_toxic_chat_instruction_2,
        empty_chat_instruction
    ],
    "detox-toxic": [
        germeval_toxic_chat_instruction, germeval_toxic_chat_instruction_1, germeval_toxic_chat_instruction_2,
        empty_chat_instruction,
    ],
    "germeval-factclaiming": [
        germeval_factclaiming_chat_instruction, germeval_factclaiming_chat_instruction_1,
        germeval_factclaiming_chat_instruction_2, empty_chat_instruction
    ],
    "germeval-engaging": [
        germeval_engaging_chat_instruction, germeval_engaging_chat_instruction_1, germeval_engaging_chat_instruction_2,
        empty_chat_instruction,
    ],
    "detox-sentiment": [
        detox_sentiment_chat_instruction, detox_sentiment_chat_instruction_1,
        detox_sentiment_chat_instruction_2, empty_chat_instruction,
    ],
    "detox-hate-speech": [
        detox_hatespeech_chat_instruction, detox_hatespeech_chat_instruction_1, detox_hatespeech_chat_instruction_2,
        empty_chat_instruction,
    ],
}

CHAT_DEMONSTRATION_QUERY = {
    "x-stance-de": [
        compose_x_stance_de_chat_demonstration_query_1, compose_x_stance_de_chat_demonstration_query_1,
        compose_x_stance_de_chat_demonstration_query_1, compose_x_stance_de_chat_demonstration_query_2,
    ],
    "detox-toxic": [
        compose_germeval_toxic_chat_demonstration_query_1, compose_germeval_toxic_chat_demonstration_query_1,
        compose_germeval_toxic_chat_demonstration_query_1, compose_germeval_toxic_chat_demonstration_query_2,
    ],
    "germeval-toxic": [
        compose_germeval_toxic_chat_demonstration_query_1, compose_germeval_toxic_chat_demonstration_query_1,
        compose_germeval_toxic_chat_demonstration_query_1, compose_germeval_toxic_chat_demonstration_query_2,
    ],
    "germeval-factclaiming": [
        compose_germeval_factclaiming_chat_demonstration_query_1,
        compose_germeval_factclaiming_chat_demonstration_query_1,
        compose_germeval_factclaiming_chat_demonstration_query_1,
        compose_germeval_factclaiming_chat_demonstration_query_2,
    ],
    "germeval-engaging": [
        compose_germeval_engaging_chat_demonstration_query_1, compose_germeval_engaging_chat_demonstration_query_1,
        compose_germeval_engaging_chat_demonstration_query_1, compose_germeval_engaging_chat_demonstration_query_2,
    ],
    "detox-sentiment": [
        compose_detox_sentiment_chat_demonstration_query_1,
        compose_detox_sentiment_chat_demonstration_query_1,
        compose_detox_sentiment_chat_demonstration_query_1,
        compose_detox_sentiment_chat_demonstration_query_2,
    ],
    "detox-hate-speech": [
        compose_detox_hatespeech_chat_demonstration_query_1, compose_detox_hatespeech_chat_demonstration_query_1,
        compose_detox_hatespeech_chat_demonstration_query_1, compose_detox_hatespeech_chat_demonstration_query_2,
    ],
}

DEMONSTRATION_QUERY = {
    "x-stance-de": [
        compose_x_stance_de_demonstration_query_1
    ],
    "germeval-toxic": [
        compose_germeval_toxic_demonstration_query_1
    ],
    "germeval-factclaiming": [
        compose_germeval_factclaiming_demonstration_query_1
    ],
    "germeval-engaging": [
        compose_germeval_engaging_demonstration_query_1
    ],
    "detox-toxic": [
        compose_germeval_toxic_demonstration_query_1
    ],
    "detox-sentiment": [
        compose_detox_sentiment_demonstration_query_1
    ],
    "detox-hate-speech": [
        compose_detox_hatespeech_demonstration_query_1
    ],
}
