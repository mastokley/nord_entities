import io
import os
import requests
import nltk
import sys

STUB = 'http://api.nordglobal.net/api'

def preprocess(text, tagset=None):
    """
    Process raw text into part-of-speech tagged sentences.

    :param text: string of raw text
    :type text: str
    :param tagset: pos tagset to use, e. g. brown, universal, wsj
    :type tagset: str
    :returns: the pos-tagged tokens
    :rtype: generator of list(tuple(str, str))

    Uses nltk's recommended off-the-shelf tagger.
    """

    # segment raw text into discrete sentences
    sentences = nltk.sent_tokenize(text)
    word_tokenized_sentences = (nltk.word_tokenize(s) for s in sentences)
    return (nltk.pos_tag(w, tagset=tagset) for w in word_tokenized_sentences)

def chunk(pos_tagged_sentence):
    """
    Chunk the sentences into groups of word tokens representing phrases

    Chunking can be rules-based or you can train a chunker with annotated data.
    In this case, I'll use the off-the-shelf, rules-based chunker provided by the
    nltk.

    Specifically, I'm using the nltk recommended "named entity" chunker.
    """
    return nltk.chunk.ne_chunk(pos_tagged_sentence)

def main(endpoint):
    # get raw text
    print('Sending HTTP request...')
    response = requests.get(''.join([STUB, endpoint]))
    if response.status_code != 200:
        raise ValueError('Invalid api endpoint.')
    print('Valid response received')
    narratives = response.json()['narratives']
    if not os.path.isdir('entities'):
        os.mkdir('entities')
    for narrative in narratives:
        filename = './entities/{}.txt'.format(narrative['id'])
        with io.open(filename, 'w') as fh:
            print('Writing file {}'.format(filename))
            for tagged_sentence in preprocess(narrative['body']):
                fh.write(str(chunk(tagged_sentence)))
    print('Done')

if __name__ == '__main__':
    try:
        main(endpoint=sys.argv[1])
    except IndexError:
        print("""Please specify api endpoint to use. For example:"""
              """\n\n\t$ python extract_entities.py """
              """'/v1/search?query=cats&sort=date&order=desc&size=10'\n""")
