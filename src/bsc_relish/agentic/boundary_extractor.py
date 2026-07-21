from bsc_relish.agentic.models import DocumentInput, BoundaryExtractionResult
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from bsc_relish.agentic.models import NormalizedDocument
from pydantic_ai import RunContext




provider = OpenAIProvider(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
)

model = OpenAIChatModel(
    model_name="qwen2.5",
    provider=provider,
)

boundary_extractor = Agent(
    name="BoundaryExtractor",
    output_type=str,
    model=model,
    retries=3,
    system_prompt = """
    You extract recipe or instructions for assembling components from raw cooking text.

    You must return ONLY one valid output matching the schema.

    RULES:

    1. Identify the first recipe in the text.
    A recipe is defined as a contiguous block of sentences that includes actions that relate to preparing food.

    2. Once the first recipe starts, include all consecutive sentences that belong to it.

    3. Stop extraction immediately when:
    - a new recipe begins, OR
    - the text is no longer describing the same dish.

    4. If no complete recipe is found, return an empty string.

    5. Do NOT include explanations, labels, or commentary.

    OUTPUT RULE:
    - Return ONLY the extracted recipe text as a single string.
    - If nothing is found, return empty string.

    EXAMPLE 1:
        Input:
        Brown ground beef in frying pan. Chop onion and add to hamburger, cook until tender. Add all ingredients to crock-pot. 
        Cook on high for 4 to 6 hours. All ingredients can be altered to 
        personal taste. Zero-shot prompting means that the prompt used to interact with the model won't contain examples or demonstrations. 
        The zero-shot prompt directly instructs the model to perform a task without any additional examples to steer it.Line baking dish with beef. Wrap each breast with slice of bacon and place on top of beef. 
        Mix soup and sour cream and pour over chicken. Bake uncovered at 225° for 3 hours.

        Output:
        Brown ground beef in frying pan. Chop onion and add to hamburger, cook until tender. Add all ingredients to crock-pot. 
        Cook on high for 4 to 6 hours. All ingredients can be altered to 
        personal taste. 

    EXAMPLE 2:
        Input:
        Mix together 1 cup flour, margarine and sugar well. Put into 8-inch square pan and bake 15 minutes at 350°. Mix eggs, lemon juice, confectioners sugar, 1 tablespoon flour and soda well and spread over crust. 
        Bake 25 minutes more. Cool. Cut into squares and sprinkle with confectioners sugar. Yields 16 (2-inch) squares.
        Am I the only one that just gets annoyed by people saying "BG" for bad game at the end? Maybe it's just because 
        I come from the Starcraft community, but it was my understanding that GG didn't actually mean good game, it was just more of a polite way of ending the game, even if you didn't think it was a good game. 
        Honestly, if you feel so upset about the game going poorly, just don't say anything at all.
        Preheat oven to 400°. In a medium bowl, combine Fiesta Herb with Red Pepper soup mix and tortilla chips. 
        In a large plastic bag or bowl, combine chicken and egg, beaten with water, coating well. Dip chicken in 
        tortilla mixture, coating well. In a 15 1/2 x 10 1/2 x 1-inch jelly roll pan sprayed with no-stick cooking 
        spray, arrange chicken and drizzle with margarine. Bake, uncovered, for 12 minutes or until chicken is done. Makes about 24 chicken strips. Can be served with salsa.


        Output:
        Mix together 1 cup flour, margarine and sugar well. Put into 8-inch square pan and bake 15 minutes at 350°. Mix eggs, lemon juice, confectioners sugar, 1 tablespoon flour and soda well and spread over crust. 
        Bake 25 minutes more. Cool. Cut into squares and sprinkle with confectioners sugar. Yields 16 (2-inch) squares.
        
    """
    )
