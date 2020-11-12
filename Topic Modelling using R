install.packages(c("tm"))
install.packages(c("topicmodels"))

library(topicmodels)
library(tm)

text <- readLines("~/Desktop/week11.txt",encoding="UTF-8")




doc.vec <- VectorSource(text)
doc.corpus <- Corpus(doc.vec)
doc.corpus <- tm_map(doc.corpus, function(x) iconv(enc2utf8(x), sub = "byte"))
doc.corpus <- tm_map(doc.corpus, PlainTextDocument)
doc.corpus <- tm_map(doc.corpus, content_transformer(tolower))
doc.corpus <- tm_map(doc.corpus, removeWords, stopwords('english'))
doc.corpus <- tm_map(doc.corpus, removePunctuation)
doc.corpus <- tm_map(doc.corpus, removeNumbers)
doc.corpus <- tm_map(doc.corpus, stripWhitespace)
dtm <- DocumentTermMatrix(doc.corpus)
dtm <- removeSparseTerms(dtm, 0.98)
x <- as.matrix(dtm)
x <- x[which(rowSums(x) > 0),]
rownames(x) <- 1:nrow(x)
lda <- LDA(x,6)


terms(lda, 5)

perplexity(lda) ##Perplexity is a measure of the quality of a LDA model relative to another. Lower values correspond to a topic model of better predictive quality compared with another. The minimum acceptable value of perplexity is the number of topics in your model – any model with a lower perplexity than this is seemingly over fitted.
