import os
from gliner import GLiNER
import pandas as pd

from bsc_relish.agentic.main_gepeto import save_report
from bsc_relish.agentic.metrics import evaluate_list
model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

text = """
HOT  CRAB  MEAT  SALAD  (continued) 

Soak  5  slices  of  bread  in  2/3  c  ,  milko   In  another  bowl  combine 
crab  meat  (thaw  first  if  frozen  crab  meat  is  used)^  eggs  that 
have  been  chopped s,  Worcestershire ^  mayonnaise^  onion ,  lemon 
juice  J  celery  flakes,  salt  and  pepper.   Toss  lightly »   Add  bread 
that  has  been  soaked  in  milk«   Mix  welL   Pour  in  shallow  casser- 
ole,  St)rinkle  top  with  parsley.    Bake  at  350  degrees  about  20 
minutes.   This  dish  may  be  made  ahead  of  time  and  kept  in  re- 
frigerator^  baking  just  before  serving.   Serves  6, 

"""
labels = ["ingredient", "action", "title", "other"]


df = pd.read_parquet("/Users/yavuzlule/Desktop/bsc-relish/src/bsc_relish/agentic/data/recipe1m_10000.parquet")
num_recipes = 100
df = df[:num_recipes]
for index, row in df.iterrows():
    text = row["directions"]
    reference_ingredients = row["NER"].split(",")  # Assuming the reference ingredients are stored as a comma-separated string
    print(text)
    entities = model.predict_entities(text, labels)

    ingredients = []
    for entity in entities: 
        if entity["label"] == "ingredient":
            ingredients.append(entity["text"])
    print("Ingredients:", ingredients)
    ingredients_report = evaluate_list(reference_ingredients, ingredients)

    base_dir = "/Users/yavuzlule/Desktop/bsc-relish/src/bsc_relish/ner"
    ingredients_path = os.path.join(base_dir, f"ingredient_extraction_gliner_{num_recipes}.json")
    save_report(ingredients_report, ingredients_path)



