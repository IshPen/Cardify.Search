from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from transformers import BertTokenizer, BertModel, pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity


# Preprocessing function
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]
    return tokens


# Embedding function
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings


# Semantic similarity function
def semantic_similarity(search_phrase, database_phrases):
    search_embedding = get_embedding(search_phrase)
    database_embeddings = [get_embedding(phrase) for phrase in database_phrases]
    similarities = [cosine_similarity(search_embedding, db_emb)[0][0] for db_emb in database_embeddings]
    return similarities


# Sentiment analysis function
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')


def get_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result['label'], result['score']


# Ranking and selection function
def find_best_match(search_phrase, database_phrases):
    search_sentiment, search_confidence = get_sentiment(search_phrase)
    print(f"Search Phrase Sentiment: {search_sentiment} (Confidence: {search_confidence:.4f})")
    similarities = semantic_similarity(search_phrase, database_phrases)

    ranked_phrases = sorted(zip(database_phrases, similarities), key=lambda x: x[1], reverse=True)
    for phrase, similarity in ranked_phrases:
        phrase_sentiment, phrase_confidence = get_sentiment(phrase)
        print(
            f"Phrase: {phrase}, Similarity: {similarity:.4f}, Sentiment: {phrase_sentiment} (Confidence: {phrase_confidence:.4f})")
        #if phrase_sentiment == search_sentiment:
        #    return phrase
    return None


# Example usage
search_phrase = "licensing good"
database_phrases = [
    "Without licensing, AI erodes trust in information, causes newspaper closure, and leads to mass job loss of journalists—that kills democracy.",
    "AI erodes search traffic for journalism",
    "Licensing helps maintain the quality and trustworthiness of information."
]

database_phrases = ['Background information', 'Tips, tricks, and navigating the file', 'The United States federal government should create licensing requirements for the use of copyrighted material for training and output of commercial generative artificial intelligence.', 'Advantage X is journalism:', 'Journalism is declining because of a perfect storm created by the unlicensed use of copyrighted material by gen AI companies—that undermines the lifeblood of US democracy', 'Unlicensed use of copyrighted content enables mass misinformation campaigns and undermines trust in all news content—congressional action is key to solve ', 'Without licensing, AI erodes trust in information, causes newspaper closure, and leads to mass job loss of journalists—that kills democracy', 'AI-driven erosion of trust in democracy in the US causes global rise in autocracy  and war', 'Global democracy collapse causes extinction', 'Copyright protections are key—it’s the only way to ensure that revenue goes to creators of content', 'Independently, lack of trust in information causes truth decay—extinction', 'The brink is now—generative AI exacerbates existing market imbalances in news media', 'Licensing is the only way to save journalism from AI', 'Licensing is feasible, even on a large scale', 'Advantage X is Model Collapse', 'Status quo AI development is unsustainable—uncompensated use of human creations causes future AI to be trained on AI-generated works, which causes model collapse and widespread AI hallucination', 'Model collapse causes large-scale data poisoning—human-generated content is key to solve', 'Specifically, model collapse makes intentional data poisoning more effective', 'And it causes widespread AI hallucinations', 'That causes escalation of military and diplomatic crises', 'AI hallucinations cause inadvertent escalation and flash wars', 'Training AI on human content is key to manage both upside and downside risks of AI', 'Managing both upside and downside risks is key to prevent extinction', 'Licensing solves—it prevents model collapse and hallucinations by creating a thriving market for original human content', 'Advantage X is Facial Recognition', 'Facial recognition technology (FRT) is outpacing regulation—swift action is key', 'Ending the “fair use” exception for training AI models constrains FRT—copyright is key because of damages', 'Limiting “fair use” is key—maintaining broad exceptions ensures proliferation of FRT', 'FRT gets equipped to weapons systems—it fundamentally transforms weapons and makes them autnomous', 'That upends strategic stability and causes nuclear escalation', 'FRT causes deepfakes—that undermines intelligence and causes first strikes', 'Deep fakes escalate—they lower the nuclear threshold, force nuclear and non-nuclear preemption, and undermine NC3', 'Empirics prove licensing is feasible and creates new revenue streams for journalism', 'Yes administration—existing and emerging CMO’s', 'Empirics prove there are tons of ways to administer licenses', 'The scale is manageable—don’t conflate ai companies wanting to ingest every work with them needing to', 'AI models are overtrained on copyrighted work now—licensing is sufficient', 'Quality matters more than quantity', 'Licensing is not too complex', 'Empirics prove the market for licenses is vast', 'Licensing works', 'Collective licensing solves and helps small publishers', 'Damle is wrong ', 'Licensing agreements are necessary for maintaining journalism and keeping an informed citizenry', 'Licensing is key to build trust in information and journalism', 'AI is the brink—it further entrenches the power of big tech companies', 'Democratic backsliding magnifies every existential threat. ', 'Democracies are comparatively more peaceful than alternatives. ', 'AI-enabled disinformation causes extinction ', 'Squo fails—licensing is narrow and AI companies are just stealing data', 'Squo doesn’t solve', 'AI developers are scraping content without permission', 'AI disrupts the market for licensing', 'Lawsuits don’t solve—they’ll rule in favor of AI, they’re case-by-case, and legislation is key', 'AI erodes search traffic for journalism', 'Generative AI is undermining journalism by taking copyrighted works without compensation', 'Licensing requirements are key to protect journalism and democracy', 'Journalism is key—only sustaining profits in the face of AI solves', 'AI undermines trust in broadcast—causes misinformation and undermines democracy', 'Humans are key to ensure trust in information', 'Strong journalism encourages more democratic participation', 'And it’s reverse casual—less journalism means less voting', 'Strong journalism solves bias', 'Local news coverage helps voters assess down-ballot candidates. Looking at people who receive information about their local elected officials compared to people who receive information about officials in neighboring states, Daniel J. Moskowitz notes that local political news coverage provides voters with “Information that allows them to assess down-ballot candidates separately from their national, partisan assessment.”', 'Our internal link is statistically verified.', 'Strong journalism is key to prevent misinformation', "It's necessary to prevent polarization", 'US leadership is key', 'Fair use fails—clarification is key', 'AI model collapse is coming now because AI is being trained on AI-created works', 'Model collapse is a likely certainty with synthetic training', 'Each generation gets worse', 'Unlicensed scraping causes model collapse—plan solves', 'Limiting fair use exceptions solves model collapse—Australia proves', 'Human generated content is key to prevent model collapse', 'Only human journalism can prevent hallucinations', 'Human training solves', 'Model collapse causes mass AI hallucinations', 'Model collapse causes large scale poisoning', 'Errors compound with every generation', 'That escalates and causes war with state and non-state actors', 'AI fueled disinformation increases the risk of nuclear escalation in a crisis', 'Model collapse turns ai profitability and sustainability', 'Copyright solves', 'Even without a complete ban, the aff solves the worst forms of FRT', 'FRT is key to LAWS development', 'FRT is integral', 'Autonomous weapons undermine deterrence', 'LAWS cause numerous existential risks', 'Licensing requirements solve deep fakes', 'Legislative action now is key', 'Deep fakes and data poisoning risk nuclear escalation', 'Miscalculation could come from non-state actors, causing catalytic nuclear war', 'Facial recognition fuels the carceral state—reject it', 'AI will collapse now—it’s a speculative bubble', 'AI is a bubble that will pop', 'Regulatory, legal, and cost challenges thump', 'No link—AI has deep pockets and other countries will also regulate AI', 'Empirics prove licensing won’t destroy the industry', "It's a normal cost of business", 'Profit margins and separate funding solve the da', 'Empirics prove licenses don’t destroy development', 'AI companies are resilient ', 'The double-bind is wrong—licensing generates revenue over time', 'The aff won’t bankrupt AI', 'The aff is negligible for investment decisions', 'Neg links are paid off', 'Open AI is lying', 'Plan increases AI value and sustainability—quality, human-made content is key', 'Without the plan, authors will hide their works—turns AI development', 'Only ethically-trained AI can sustain the industry', 'Strong copyright protections are key build demand for AI systems', 'Diverse, human-created works are key to effective models', 'Strong copyright protections are comparatively more important for US competitiveness', 'Unlicensed AI development undermines human creativity', 'Human creativity is key to breakthrough innovations—solves better than AI', 'Licensing solves innovation and economic competitiveness', 'Only strong copyright protections solves US competitiveness', 'Only human creativity creates breakthrough innovations', 'AI fails—data plateaus and not enough electricity', 'AI can’t solve social problems', 'AI isn’t a better decision-maker for complicated problems', 'Case outweighs on timeframe and probability', 'A lack of innovation and adoption of A.I. is slowing military integration, but further innovation could revolutionize military technology.', 'Effective military A.I. enables a credible first-strike capability, creating use-it-or-lose-it pressures and pushing adversaries towards launch-on-warning postures. This ensures accidental wars and nuclear escalation.', 'AI innovation gets integrated into the military. That causes insurmountable gaps in nuclear power, upending MAD and decimating strategic stability.', 'Increased private sector A.I. innovation will be funneled into the military.', 'Deterrence solves every AFF impact. Only military A.I. upends 70 years of stability.', 'Military A.I. causes Russia launch-on-warning, triggering accidental nuclear war.', 'Improved military A.I. pushes China to high alert, guaranteeing nuclear escalation.', 'Humans won’t detect failures---they’ll over-trust military A.I., prompting escalation.', 'AI innovation causes extinction', 'AI causes extinction in a myriad of ways', 'Specifically, misalignment causes extinction', 'Or weaponization or accidents', 'It could become completely uncontrollable', 'AI doesn’t have to be malicious to cause extinction', 'Continuing AI development ensures extinction—a pause is key', 'Aligning AI doesn’t solve extinction—pause is key', 'AGI is possible', 'It’s not too far off', 'Science research and innovation are declining', 'No link—restricting commercial ai uses doesn’t spill over to TDM for research', 'No link—TDM research can use licensed works—existing markets prove', 'Licensing is frequently used now for scientific research', '“Fair use” doesn’t solve research—it’s just as unpredictable', 'Research isn’t key to scientific progress or innovation', 'Research isn’t key—it overstretches scientists', 'Bogus papers undermine the application and trust of research', 'No impact—science doesn’t prevent extinction', 'No science impact—it isn’t reflected by policy choices', 'Copyright suits will flood the courts now, but the plan provides legal clarity which solves', 'Licensing creates partnership between authors and ai developers', 'It removes the threat of litigation', 'And insulates companies from legal risk', 'There’s only a risk of the turn—unlicensed use creates serial lawsuits', "It's already happening", 'CP causes uncertainty and undermines aff solvency', 'Copyright licensing is comparatively better—tech changes too fast for sui generis to be appropriate', 'Formal licensing is key', 'Opt-out fails—piracy, lack of tech, and non-compliance', 'Developers will sabotage opt-out procedures', 'The cp empirically fails and puts the onus on publishers—can’t solve', 'The cp is wholly insufficient', 'Only the plan solves—too many technical barriers to opt-outs', 'It cuts off search traffic to publishers', 'Err heavily aff—the cp is a big-tech ruse', 'Robots.txt fails', 'Can’t solve either advantage—delay ', 'Media companies would shutter before litigation was decided', 'Delay deficit to courts', 'Congress is key to certainty', 'Congress key—actions must be explicit and statutory', 'Explicit regulation is key', 'States get preempted', 'CP undermines publishers and causes political interference in the media', 'It would drastically under-compensate publishers', 'It has high costs and undermines publishers rights']

best_match = find_best_match(search_phrase, database_phrases)
print(f"Best Match: {best_match}")


