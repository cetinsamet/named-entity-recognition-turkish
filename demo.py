from sklearn_crfsuite import CRF
import jpype as jp
import argparse
import pickle


ZEMBEREK_PATH	= 'bin/zemberek-full.jar'
MODEL_PATH 		= "model/crf08.pickle"

def getFeature(token, token_index, sentence):

	feature = {	'token'             : token,
				'is_first'          : token_index == 0,
				'is_last'           : token_index == len(sentence) - 1,

				'is_capitalized'    : token[0].upper() == token[0],
				'is_all_capitalized': token.upper() == token,
				'is_capitals_inside': token[1:].lower() != token[1:],
				'is_numeric'        : token.isdigit(),
				'is_numeric_inside' : any([c.isdigit() for c in token]),
				'is_alphanumeric'   : token.isalnum(),

				'prefix-1'          : token[0],
				'suffix-1'          : token[-1],

				'prefix-2'          : '' if len(token) < 2  else token[:2],
				'suffix-2'          : '' if len(token) < 2  else token[-2:],

				'prefix-3'          : '' if len(token) < 3  else token[:3],
				'suffix-3'          : '' if len(token) < 3  else token[-3:],

				'prefix-4'          : '' if len(token) < 4  else token[:4],
				'suffix-4'          : '' if len(token) < 4  else token[-4:],

				'prev-token'        : '' if token_index == 0 else sentence[token_index - 1],
				'next-token'        : '' if token_index == len(sentence) - 1 else sentence[token_index + 1],

				'2-prev-token'      : '' if token_index <= 1 else sentence[token_index - 2],
				'2-next-token'      : '' if token_index >= len(sentence) - 2 else sentence[token_index + 2],
	}

	return feature


def main(sentence):

	# Load best conditional random fields model
	with open(MODEL_PATH, 'rb') as infile:
		crf = pickle.load(infile)

	# Start the JVM
	jp.startJVM(jp.getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH))
	tokenizer 	= jp.JClass('zemberek.tokenization.TurkishTokenizer').DEFAULT


	tokenized_sentence = []
	tokenIterator = tokenizer.getTokenIterator(sentence)

	while (tokenIterator.hasNext()):
		tokenized_sentence.append(tokenIterator.token.getText())

	features = [getFeature(word, i, tokenized_sentence) for i, word in enumerate(tokenized_sentence)]
	labels 	 = crf.predict([features])

	for word, label in list(zip(tokenized_sentence, labels[0])):
		print("%-15s : %s" % (word, label))

	return


if __name__ == '__main__':


	parser = argparse.ArgumentParser()
	parser.add_argument("-s", "--sentence",
						help="write an input sentence to be labeled")

	args 		= parser.parse_args()
	sentence 	= args.sentence
	
	main(sentence)