import argparse
import io
import requests
import nltk

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


def main(endpoint, outfile=None):
    url = ''.join([STUB, endpoint])
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError('Invalid api endpoint.')
    narratives = response.json()['narratives']
    for narrative in narratives:
        out = ''
        for tagged_sentence in preprocess(narrative['body']):
            out += str(chunk(tagged_sentence))
    if outfile:
        with io.open(outfile, 'w') as fh:
            fh.write(out)
            print('Wrote {}'.format(outfile))
    else:
        print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('endpoint',
                        help="specifies api endpoint, e.g. '/v1/search?query=cats'")
    parser.add_argument('-w',
                        '--write',
                        help='writes output to file instead of stdout')
    args = parser.parse_args()

    main(args.endpoint, outfile=args.write)
