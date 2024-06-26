HYPERPARAMS = {
    "detox-sentiment":{
        "deepset/gbert-base":{
            "batch_size": 8,
            "learning_rate": 1e-05
        },
        "bert-base-multilingual-cased":{
            "batch_size": 16,
            "learning_rate": 1e-05
        },
        "bert-base-uncased":{
            "batch_size": 32,
            "learning_rate": 1e-05
        },
        "deepset/gbert-large":{
            "batch_size": 16,
            "learning_rate": 1e-05
        },
        "deepset/gelectra-base":{
            "batch_size": 8,
            "learning_rate": 1e-05
        },
        "deepset/gelectra-large":{
            "batch_size": 16,
            "learning_rate": 1e-05
        },
        "microsoft/mdeberta-v3-base":{
            "batch_size": 8,
            "learning_rate": 1e-05
        },
        "FacebookAI/xlm-roberta-base":{
            "batch_size": 8,
            "learning_rate": 1e-05
        },
        "roberta-base":{
            "batch_size": 32,
            "learning_rate": 1e-05
        },
        "microsoft/deberta-v3-base":{
            "batch_size": 8,
            "learning_rate": 1e-05
        },
    },
    "detox-hate-speech":{
        "deepset/gbert-base":{
            "batch_size": 16,
            "learning_rate": 5e-05
        },
        "bert-base-multilingual-cased":{
            "batch_size": 8,
            "learning_rate": 2e-05
        },
        "bert-base-uncased":{
            "batch_size": 16,
            "learning_rate": 2e-05
        },
        "deepset/gbert-large":{
            "batch_size": 32,
            "learning_rate": 2e-05
        },
        "deepset/gelectra-base":{
            "batch_size": 16,
            "learning_rate": 2e-05
        },
        "deepset/gelectra-large":{
            "batch_size": 16,
            "learning_rate": 2e-05
        },
        "microsoft/mdeberta-v3-base":{
            "batch_size": 8,
            "learning_rate": 1e-05
        },
        "FacebookAI/xlm-roberta-base":{
            "batch_size": 32,
            "learning_rate": 2e-05
        },
        "roberta-base":{
            "batch_size": 16,
            "learning_rate": 1e-05
        },
        "microsoft/deberta-v3-base":{
            "batch_size": 8,
            "learning_rate": 2e-05
        },
    },
    "detox-toxic":{

        "deepset/gbert-base":{
            "batch_size": 8,
            "learning_rate": 1e-05
        },
        "bert-base-multilingual-cased":{
            "batch_size": 16,
            "learning_rate": 1e-05
        },
        "bert-base-uncased":{
            "batch_size": 16,
            "learning_rate": 1e-05
        },
        "deepset/gbert-large":{
            "batch_size": 16,
            "learning_rate": 1e-05
        },
        "deepset/gelectra-base":{
            "batch_size": 16,
            "learning_rate": 5e-05
        },
        "deepset/gelectra-large":{
            "batch_size": 32,
            "learning_rate": 5e-05
        },
        "microsoft/mdeberta-v3-base":{
            "batch_size": 16,
            "learning_rate": 1e-05
        },
        "FacebookAI/xlm-roberta-base":{
            "batch_size": 32,
            "learning_rate": 2e-05
        },
        "roberta-base":{
            "batch_size": 16,
            "learning_rate": 2e-05
        },
        "microsoft/deberta-v3-base":{
            "batch_size": 16,
            "learning_rate": 1e-05
        },
    },
    "x-stance-de":{
        "deepset/gbert-base":{
            "batch_size": 32,
            "learning_rate": 5e-05
        },
        "bert-base-multilingual-cased":{
            "batch_size": 16,
            "learning_rate": 2e-05
        },
        "bert-base-uncased":{
            "batch_size": 16,
            "learning_rate": 2e-05
        },
        "deepset/gbert-large":{
            "batch_size": 16,
            "learning_rate": 1e-05
        },
        "deepset/gelectra-base":{
            "batch_size": 16,
            "learning_rate": 2e-05
        },
        "deepset/gelectra-large":{
            "batch_size": 32,
            "learning_rate": 1e-05
        },
        "FacebookAI/xlm-roberta-base":{
            "batch_size": 32,
            "learning_rate": 2e-05
        },
        "roberta-base":{
            "batch_size": 16,
            "learning_rate": 1e-05
        },
        "microsoft/mdeberta-v3-base":{
            "batch_size": 32,
            "learning_rate": 2e-05
        },
        "microsoft/deberta-v3-base":{
            "batch_size": 32,
            "learning_rate": 2e-05
        },
    },
    "germeval-factclaiming":{

        "deepset/gbert-base":{
            "batch_size": 16,
            "learning_rate": 5e-05
        },
        "bert-base-multilingual-cased":{
            "batch_size": 16,
            "learning_rate": 2e-05
        },
        "bert-base-uncased":{
            "batch_size": 16,
            "learning_rate": 5e-05
        },
        "deepset/gbert-large":{
            "batch_size": 16,
            "learning_rate": 2e-05
        },
        "deepset/gelectra-base":{
            "batch_size": 32,
            "learning_rate": 5e-05
        },
        "deepset/gelectra-large":{
            "batch_size": 16,
            "learning_rate": 2e-05
        },
        "FacebookAI/xlm-roberta-base":{
            "batch_size": 16,
            "learning_rate": 2e-05
        },
        "roberta-base":{
            "batch_size": 8,
            "learning_rate": 2e-05
        },
        "microsoft/mdeberta-v3-base":{
            "batch_size": 32,
            "learning_rate": 2e-05
        },
        "microsoft/deberta-v3-base":{
            "batch_size": 32,
            "learning_rate": 5e-05
        },
    },
    "germeval-engaging":{

        "deepset/gbert-base":{
            "batch_size": 16,
            "learning_rate": 5e-05
        },
        "bert-base-multilingual-cased":{
            "batch_size": 32,
            "learning_rate": 1e-05
        },
        "bert-base-uncased":{
            "batch_size": 8,
            "learning_rate": 5e-05
        },
        "deepset/gbert-large":{
            "batch_size": 32,
            "learning_rate": 1e-05
        },
        "deepset/gelectra-base":{
            "batch_size": 32,
            "learning_rate": 2e-05
        },
        "deepset/gelectra-large":{
            "batch_size": 32,
            "learning_rate": 1e-05
        },
        "FacebookAI/xlm-roberta-base":{
            "batch_size": 32,
            "learning_rate": 5e-05
        },
        "roberta-base":{
            "batch_size": 16,
            "learning_rate": 2e-05
        },
        "microsoft/mdeberta-v3-base":{
            "batch_size": 16,
            "learning_rate": 2e-05
        },
        "microsoft/deberta-v3-base":{
            "batch_size": 32,
            "learning_rate": 2e-05
        },
    },
    "germeval-toxic":{
        "deepset/gbert-base":{
            "batch_size": 16,
            "learning_rate": 5e-05
        },
        "bert-base-multilingual-cased":{
            "batch_size": 8,
            "learning_rate": 1e-05
        },
        "bert-base-uncased":{
            "batch_size": 8,
            "learning_rate": 5e-05
        },
        "deepset/gbert-large":{
            "batch_size": 16,
            "learning_rate": 1e-05
        },
        "deepset/gelectra-base":{
            "batch_size": 16,
            "learning_rate": 5e-05
        },
        "deepset/gelectra-large":{
            "batch_size": 32,
            "learning_rate": 5e-05
        },
        "FacebookAI/xlm-roberta-base":{
            "batch_size": 32,
            "learning_rate": 5e-05
        },
        "roberta-base":{
            "batch_size": 16,
            "learning_rate": 1e-05
        },
        "microsoft/mdeberta-v3-base":{
            "batch_size": 16,
            "learning_rate": 2e-05
        },
        "microsoft/deberta-v3-base":{
            "batch_size": 8,
            "learning_rate": 2e-05
        },
    },
}