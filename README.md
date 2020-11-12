# Social-Media-Analytics using Python. 

Twitter - Elon Musk

Elon Musk is chosen as a famous person of interest and related five key terms are used for gathering more relevant tweets.

This repository shows data collection using Standard API - REST and highlights pre-processing steps before moving on to data analysis. The first step in data collection involved selecting appropriate key terms for extraction of tweets to create an investigative corpus. The key terms selected as previously mentioned are SpaceX, SolarCity, Tesla, Starlink and Rockets. In this growing digital world, people have learned to communicate via social media and express their mindset increasingly well enough to perform distinctive exploratory analysis. People express their life events quite comfortably every day in unusual or usual ways and nowadays popularity has gained in showcasing their feelings with various categorical way such as emojis, GIFs, videos, images but pre-dominantly text. For analysts to comprehensively yield benefits from analysing such surge of communication requires necessary data pre-processing such as removing stop words, abbreviations, URLs, special characters, spell checking and more, as shown in the python code file. 

For data analysis, general questions have been answered and graphically revealed. It focuses on identifying the count of the most commonly used words in the dataset and converted first into a table format and then a graph. Also, number of tweets posted per day, number of unique users per day, top 10 most active users over the entire period. Another approach of visualising cleaned text is through the use of word cloud. Word cloud, also known as tag or concept cloud is a visual representation to depict keyword metadata or tags. Most commonly single words are represented and the importance of each word is shown with font size or colour. The format is useful in rapidly perceiving the most prominent key terms and for identifying its relative prominence.

Lastly, sentiment modelling is carried out where each step is outlined along with the results produced. 

Addtionally, topic modelling is conducted using R programming language as shown in a separate file. Topic modelling is a kind of statistical modelling where the discovery of abstract ‘topics’ within a collection of text data or documents is made. It is a process where documents are analysed to discover in-depth insights. Latent Dirichlet Allocation (LDA) as an example of the topic model is a mechanism used to classify text in a document to a particular topic. It calculates a topic per document model and words per topic model, also known as Dirichlet distributions. In LDA, each word or phrase is linked to ‘k’ and perplexity is obtained from it which is a measure of how well a model predicts the document provided. This allows in uncovering latent semantic structures within the document. Lower values are mostly preferred and values less than a number of topics implies over-fitting. For finer observation, text pre-processing is recommended where punctuation and stop words are excluded from the document. 

The following intuition is behind successfully implementing a topic model:
1. First, a number of topics: k was selected.
2. All possible words: w(vocab) were collected.
3. For every tweet, each word was assigned to a topic.
4. Finally, approximate distribution of all words across each topic results were obtained

There were few steps there were carried out in R Studio as shown in the code file. After calling out the required topicmodels package, the tweets csv file was read by specifying a local file path. Next, the text file underwent the same procedure as it did while using Python i.e. data pre-processing such as conversion to lowercase, removal of stop words, punctuation marks, numbers and whitespace. After this, very rare terms (occurring in at least 2% of the tweets) were removed. Moreover, tweets that did not contain one of the remaining terms were removed. Finally, a document matrix was created out of corpus, the number of topic models was specified (5) and LDA was calculated as shown below. 

