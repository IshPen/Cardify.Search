# Cardify.Search
Semantic Searching for Debate Evidence in Document Taglines

Runnable Streamlit app is KNN_APP.py -- run ```streamlit run KNN_APP.py``` in command prompt to run

More info in this [Slideshow](https://www.canva.com/design/DAGLyI8GMU8/UxsBu1HPkfBHyq8xHMajEA/edit?utm_content=DAGLyI8GMU8&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

Governor's Science and Technology Champions' Academy '24 @SMU

## Description
Cardify.Search is a novel search tool that combines both _literal distancing_ and _semantic(sentiment-based) distancing_ between phrases.

 - Literal Distance - The distance between the literal letters, characters, or substrings between two words or phrases. Can be used to tell how similar two words *look*.
 - Semantic Distance - The fun one, uses the embedded definitions of words to determine whether two phrases are semantically similar.
 
 Cardify.Search combines both of these distance measures to determine to create a search algorithm for highschool + college debate evidence finding.

## How would this help me?
Debate evidence files are reaaaaaallllly biiiigggg... and naturally so. Debaters have a lot of arguments and positions they need to prep for, warranting a lot of offensive and defensive evidence. It doesn't help that debaters are time-crunched in round, leading to easily using the wrong evidence or not knowing where the correct one is since you keep CTRL-Fing the wrong term.

For instance, CTRL-Fing "Facial Recognition" when the evidence you're looking for is actually labelled "Deep Fake Tech" doesn't show you where the correct evidence you need is. With Cardify.Search's Semantic Distance search dimension however, facial recognition and deepfake are _semantically similar_, meaning although they don't have the same literal words, the program still considers it similar due to it's definition. [(example of this on slide 5)](https://www.canva.com/design/DAGLyI8GMU8/UxsBu1HPkfBHyq8xHMajEA/edit?utm_content=DAGLyI8GMU8&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

This makes sorting through evidence files significantly easier and smarter, although an actual integration into Microsoft Word/Verbatim would probably be much better.
## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`DOCX_PATH_FILE` -- this is the .docx file (Microsoft Word) containing all of your debate evidence (AKA: the master file) - the program uses the paragraphs with @Heading Level 3 to find "blocks"/sections of debate evidence in your master file 

`OPENAI_API_KEY` -- this is just used for artificial data creation of search terms and their corresponding taglines in "create_phrase_for_heading.py". Not important if you're only running KNN_APP.

## Sections

### Experiments
Contains my experiments on tagline lemmatizations and data normalization

### Literal Distance
Holds two different types of literal distance calculators for the app -- traditional Levenstein Distance and a custom literal distance calculator which was ended up being used in the final program

### Sentiment Based
Different experiments for sentiment-based distance calculators for the app -- the cosine similarity distance calculator utilizing the 768-dimension BERT embedding model was used to get pairwise similarity between terms (see also reconstructed_pairwise_cosine_similarity.py)


