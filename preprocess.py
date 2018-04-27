import argparse
import nltk


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-file-path', type=str)
    parser.add_argument('--tgt-file-path', type=str)
    return parser.parse_args()


def lower(sent):
    return sent.lower()


def tokenize(sent):
    return ' '.join(nltk.word_tokenize(sent))


def is_too_long(src_sent, tgt_sent, n=20):
    return len(src_sent.split()) >= n or len(tgt_sent.split()) >= n


def is_ratio_unbalance(src_sent, tgt_sent, ratio=1.5):
    src_len = len(src_sent.split())
    tgt_len = len(tgt_sent.split())
    return (src_len / tgt_len) > ratio


def if_too_short(src_sent, tgt_sent, n=3):
    return len(src_sent.split()) <= n or len(tgt_sent.split()) <= n


def run_all(sent):
    sent = tokenize(sent)
    sent = lower(sent)
    return sent


def is_starts_with(src_sent, tgt_sent, tokens=['<']):
    for token in tokens:
        if src_sent.startswith(token) or tgt_sent.startswith(token):
            return True
    return False


def main():
    args = get_args()
    src_in_f = open(args.src_file_path, 'r')
    tgt_in_f = open(args.tgt_file_path, 'r')
    src_out_f = open(args.src_file_path + '.prepared', 'a')
    tgt_out_f = open(args.tgt_file_path + '.prepared', 'a')

    src_in_line = src_in_f.readline().strip()
    tgt_in_line = tgt_in_f.readline().strip()

    src_counter = 0
    tgt_counter = 0

    while src_in_line and tgt_in_line:
        src_in_line_preped = run_all(src_in_line)
        tgt_in_line_preped = run_all(tgt_in_line)

        if (is_too_long(src_in_line_preped, tgt_in_line_preped) or
           is_ratio_unbalance(src_in_line_preped, tgt_in_line_preped)):
            pass
        else:
            # write
            src_out_f.write(src_in_line_preped)
            src_out_f.write('\n')
            src_counter += 1

            tgt_out_f.write(tgt_in_line_preped)
            tgt_out_f.write('\n')
            tgt_counter += 1

        src_in_line = src_in_f.readline().strip()
        tgt_in_line = tgt_in_f.readline().strip()

    print('%s lines for source data' % src_counter)
    print('%s lines for target data' % tgt_counter)


if __name__ == '__main__':
    main()
