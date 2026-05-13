from typing import List, Dict, Callable, Iterable
from dataclasses import dataclass

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from safetensors.torch import load_file

from bsc_relish import infer_bert
from bsc_relish.infer_roberta import get_device


# ------------------------------------------------------------
# DATA STRUCTURES
# ------------------------------------------------------------

@dataclass
class WindowPrediction:
    start_token: int
    end_token: int
    window_size: int
    text: str
    score: float
    label: str


# ------------------------------------------------------------
# TOKENIZATION
# ------------------------------------------------------------

def whitespace_tokenize(text: str) -> List[str]:
    """
    Simple tokenizer.
    Replace with a transformer tokenizer if needed.
    """
    return text.split()


def detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)


# ------------------------------------------------------------
# SLIDING WINDOW GENERATION
# ------------------------------------------------------------

def sliding_windows(
    tokens: List[str],
    min_window: int = 2,
    max_window: int = 6,
    stride: int = 1,
) -> Iterable[Dict]:
    """
    Generate all sliding windows between min_window and max_window.

    Example:
        tokens = ["The", "Lord", "of", "the", "Rings"]

        window_size=2:
            ["The Lord"]
            ["Lord of"]
            ["of the"]
            ["the Rings"]

        window_size=3:
            ["The Lord of"]
            ...
    """

    n = len(tokens)

    for window_size in range(min_window, max_window + 1):

        if window_size > n:
            continue

        for start in range(0, n - window_size + 1, stride):

            end = start + window_size

            yield {
                "start": start,
                "end": end,
                "window_size": window_size,
                "tokens": tokens[start:end],
            }


# ------------------------------------------------------------
# CLASSIFIER INTERFACE
# ------------------------------------------------------------

def classifier(text: str, model, tokenizer, device, max_length) -> Dict:
    """
    Mock classifier.

    Replace with your real model inference.

    Expected return format:
    {
        "label": "TITLE" or "NOT_TITLE",
        "score": probability/confidence
    }
    """

    print("Loading model and tokenizer...")
    
        
    try:
        prob = infer_bert.predict_single_text(text, model, tokenizer, device, max_length)
        return {
                    "label": "TITLE" if prob > 0.5 else "NOT_TITLE",
                    "score": prob
                }
    except Exception as e:
        print(f"  Error processing text {e}")

# ------------------------------------------------------------
# INFERENCE PIPELINE
# ------------------------------------------------------------

def detect_titles(
    model,
    tokenizer,
    text: str,
    min_window: int = 2,
    max_window: int = 6,
    stride: int = 1,
    threshold: float = 0.5,
) -> List[WindowPrediction]:
    """
    Run classifier over all token windows.
    """
    tokens = whitespace_tokenize(text)
    predictions = []


    for window in sliding_windows( 
        tokens,
        min_window=min_window,
        max_window=max_window,
        stride=stride,
    ):

        window_text = detokenize(window["tokens"])
        
        prob = infer_bert.predict_single_text(window_text, model, tokenizer, device, 32)
        result = {
            "label": "TITLE" if prob > 0.5 else "NOT_TITLE",
            "score": prob
        }
        if (
            result["label"] == "TITLE"
            and result["score"] >= threshold
        ):

            predictions.append(
                WindowPrediction(
                    start_token=window["start"],
                    end_token=window["end"],
                    window_size=window["window_size"],
                    text=window_text,
                    score=result["score"],
                    label=result["label"],
                )
            )

    return predictions


# ------------------------------------------------------------
# OPTIONAL: NON-MAX SUPPRESSION / DEDUP
# ------------------------------------------------------------

def remove_overlapping_predictions(
    predictions: List[WindowPrediction]
) -> List[WindowPrediction]:
    """
    Keep highest-scoring overlapping spans.
    """

    predictions = sorted(
        predictions,
        key=lambda x: x.score,
        reverse=True,
    )

    selected = []

    for pred in predictions:

        overlaps = False

        for existing in selected:

            overlap = not (
                pred.end_token <= existing.start_token
                or pred.start_token >= existing.end_token
            )

            if overlap:
                overlaps = True
                break

        if not overlaps:
            selected.append(pred)

    return sorted(selected, key=lambda x: x.start_token)

def print_top_n(predictions: list[WindowPrediction], n: int = 5):
    top_preds = sorted(
        predictions,
        key=lambda p: p.score,
        reverse=True
    )[:n]

    for i, pred in enumerate(top_preds, 1):
        print(
            f"{i}. score={pred.score:.4f} "
            f"label={pred.label} "
            f"text={pred.text}"
        )

# ------------------------------------------------------------
# EXAMPLE
# ------------------------------------------------------------

if __name__ == "__main__":
    
    text = """
            one finger high and four fingers wide, chop up fine with a piece of onion, piece of celery, piece of carrot, and put into a saucepan. Take three-quarters of a pound of meat, either lamb, veal, beef, or fresh pork, cut it into several pieces, 56 salt and pepper it, and put a pinch of allspice, then put it into the saucepan; cook it until it is well colored, then add two tablespoons of red or white wine. When it is absorbed add one tablespoon of tomato paste, dissolved in water, or tomato sauce of fresh tomatoes (receipt Tomato Sauce No. i). Cook over a moderate fire, one hour longer if the meat is veal or lamb, and one and one-half hours to two hours for pork or beef, adding water if necessary. This meat can be served with Ribbon Macaroni. Put the meat in the middle, the macaroni around it, and the sauce over all, adding two tablespoons of grated Parmesan cheese to the macaroni after it is boiled, and mixing well before putting it on the platter. Sprinkle on a little more cheese before carrying to the table. This dish can be made equally well with left-over meats of any kind, turkey being especially good served this way. Salad "del Prevosto" Boil in their skins three good-sized potatoes, peel them and slice them, then put them into a salad bowl, and pour over them one-half a glass of white wine. Do this about two or three hours before they are wanted, so the potatoes
            will have time thoroughly to absorb the wine. From time to time mix them with a fork and spoon to let the wine permeate. A few minutes before the meal make a good French salad dressing, add some pickled peppers cut up, some capers, and some chopped-up parsley, pour on the French dressing, mix up well, and serve. The Cardinal's Salad Wash a good lettuce and a bunch of water-cress. Cut a cold boiled beef into strips, add six radishes, two hard-boiled eggs chopped up, and one small sliced cucumber. Arrange the lettuce-leaves in a salad-bowl, mix the other ingredients with a sufficient quantity of mayonnaise sauce, put them in the midst of the lettuce, and serve. Take a head of endive, wash it and dry it well, and put it into a salad-bowl. Pour over it three table- spoons of good olive-oil. Mix one tablespoon of 58 honey (or sugar), one of vinegar, and salt and pepper in a cup, and pour over the salad just before serving. Cut one carrot and one turnip into slices, and cook them in boiling soup. When cold, mix them with two cold boiled potatoes and one beet cut into strips. Add a very little chopped leeks or onion, pour some sauce, "Lombarda" (see Sauces, page 31), over the salad, and garnish with water-cress. Chop up six lettuce-leaves and three stalks of celery, cut up the remains of a cold fowl in small pieces, and mix with one tablespoon of vinegar and salt and pepper in a salad bowl.
            Pour a cup of mayonnaise sauce over, and garnish with quarters of hard-boiled egg, one tablespoon of capers, six stoned olives, and some small, tender lettuce-leaves. Cut into small pieces one cold boiled beet and half an onion. Add some cold boiled string-beans, some cold boiled asparagus tips, two tablespoons of cold cooked peas, one cold boiled carrot, and some celery. Mix them together, and pour over all a mayonnaise sauce. Add the juice of a lemon and serve. Take twenty good chestnuts and roast them on a slow fire so that they won't color. Remove the shells without breaking the nuts, and put them into a sauce- pan with one level tablespoon of powdered sugar and one-half glass of milk and a little vanilla. Cover the saucepan and let it cook slowly (simmer) for more than a half -hour. Then drain the chestnuts and pass them through a sieve. Put them back in a bowl with one-half a tablespoon of butter, the yolks of three eggs, and mix well without cooking. Allow them to cool, and then take a small portion at a time, the size of a nut, roll them, dip them in egg, and in bread crumbs, and fry in butter and lard, a few at a time. Serve hot with powdered sugar. Chestnuts "alia Lucifero " Take forty good chestnuts and roast them over a slow fire. Do not allow them to become dried up or colored. Remove the shells carefully, put them in a bowl, and pour over them one -half
    """

    text_2 = """


            No-Bake Nut Cookies, 1 c. firmly packed brown sugar, 1/2 c. evaporated milk, 1/2 tsp. vanilla, 1/2 c. broken nuts..., Creamy Corn, Combine first four ingredients and press in 13 x 9-inch ungreased pan. Melt chocolate chips and..., Rhubarb Coffee Cake


            """
    model_path = "/Users/yavuzlule/Desktop/bsc-relish/results/distilbert-base-cased/2026-05-12_16-00-12/model.safetensors"
    
    
    model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-cased',
    num_labels=2  # binary classification
    )
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")

    state_dict = load_file(model_path)
    
    # Convert TensorFlow naming to PyTorch
    print("Converting TensorFlow parameters to PyTorch format...")
    state_dict = infer_bert.convert_tf_to_pytorch_layernorm(state_dict)
    
    # Load with strict=False to handle any remaining mismatches
    model.load_state_dict(state_dict, strict=False)
    
    # Set to evaluation mode
    model.eval()
    print("Model loaded successfully!")
    device = get_device()
    model.to(device)

    raw_predictions = detect_titles(
        model= model,
        tokenizer=tokenizer,
        text=text_2,
        min_window=2,
        max_window=5,
        stride=1,
        threshold=0.9995,
    )

    final_predictions = remove_overlapping_predictions(
        raw_predictions
    )

    print_top_n(final_predictions, n=10)