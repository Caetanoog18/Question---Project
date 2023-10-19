import nltk
#nltk.download('stopwords')
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    
    file_name = dict()

    # Iterate over the files in the specified directory.
    for file in os.listdir(directory):
        # Open each file in read ('r') mode with utf-8 encoding.
        with open(os.path.join(directory, file), encoding='utf-8') as file_n:
            # Read the content of the file as a string and store it in the dictionary.
            file_name[file]= file_n.read()

    return file_name
    


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    
    words = nltk.word_tokenize(document.lower())

    # Removing any punctuation
    words = [word for word in words if word not in string.punctuation]


    # Removing stopwords in english
    stop_words = set(nltk.corpus.stopwords.words("english"))
    words = [word for word in words if word not in stop_words]


    return words



def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    
    # Initialize an empty dictionary to store IDF values
    idf_values = {}
    
    # Loop through each document
    total_documents = len(documents)
    for document_name, words in documents.items():
        # Create a set of unique words in the document
        unique_words = set(words)
        
        # Update word counts in the IDF dictionary
        for word in unique_words:
            if word in idf_values:
                idf_values[word] += 1
            else:
                idf_values[word] = 1
    
    # Calculate IDF values
    for word, count in idf_values.items():
        idf_values[word] = math.log(total_documents / count)
    
    # Return the IDF dictionary
    return idf_values


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    
    # Create a list to store (filename, tf-idf score) tuples
    file_scores = []

    for filename, words_in_file in files.items():
        # Calculate the tf-idf score for each word in the query
        tf_idf_score = 0
        for word in query:
            if word in words_in_file and word in idfs:
                tf = words_in_file.count(word)
                tf_idf_score += tf * idfs[word]

        # Append the (filename, tf-idf score) tuple to the list
        file_scores.append((filename, tf_idf_score))

    # Sort the list of (filename, tf-idf score) tuples by tf-idf score in descending order
    file_scores.sort(key=lambda x: x[1], reverse=True)

    # Extract the filenames from the sorted list
    top_n_files = [filename for filename, _ in file_scores[:n]]

    return top_n_files

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # Create a list to store tuples of (sentence, matching word measure, query term density)
    ranked_sentences = []

    # Iterate through each sentence
    for sentence, words_in_sentence in sentences.items():
        # Calculate matching word measure for the sentence
        matching_word_measure = sum(idfs[word] for word in query if word in words_in_sentence)
        
        # Calculate query term density for the sentence
        query_term_density = sum(1 for word in words_in_sentence if word in query) / len(words_in_sentence)
        
        # Append the sentence along with its matching word measure and query term density to the list
        ranked_sentences.append((sentence, matching_word_measure, query_term_density))
    
    # Sort the sentences based on matching word measure (in descending order) and query term density (in descending order)
    ranked_sentences.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    # Get the top n sentences from the sorted list
    top_n_sentences = [sentence for sentence, _, _ in ranked_sentences[:n]]
    
    return top_n_sentences

if __name__ == "__main__":
    main()