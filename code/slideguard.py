"""Assert the reporting rules that had no mechanism, before they fail.

Two rules were written in prose and never enforced:

  * every per-object figure carries all three identities -- DESI target,
    Opus 5, Redrock -- never two of the three.  The target bit is the only one
    independent of both reader and reference, and without it two failures of
    the same kind read as unrelated.
  * the reader is named "Opus 5".  Not "AI student", not "the model", not
    "mine".  The page compares two named pipelines; a role noun makes it read
    as a demo rather than a measurement.

Both are checked here rather than remembered.  Import into any slide generator
and call check_text() on each subtitle before rendering.

Written pre-emptively: the previous three rule failures each got a mechanism
only after breaking.  These two have not broken yet.
"""
import re
import sys

BANNED = ['AI student', 'the model', 'the AI', 'my answer:']
REQUIRED = ['DESI target', 'Opus 5', 'Redrock']


def check_text(text, where='slide', require_identity=True):
    """Raise SystemExit on a violation.  `where` names the caller for the msg."""
    bad = [b for b in BANNED if b.lower() in text.lower()]
    if bad:
        raise SystemExit('%s: forbidden label(s) %s -- the reader is named '
                         '"Opus 5".' % (where, ', '.join(repr(b) for b in bad)))
    if require_identity:
        miss = [r for r in REQUIRED if r not in text]
        if miss:
            raise SystemExit(
                '%s: a per-object figure must carry all three identities; '
                'missing %s. Full form: "DESI target: ELG   Opus 5: BGS, '
                'z = 0.3809   Redrock: GALAXY, z = 1.6739 (ZWARN 4)"'
                % (where, ', '.join(miss)))
    return text


def check_confusion(title, M, where='matrix'):
    """Refuse a confusion-matrix caption that quotes a score the matrix does
    not show.

    Correction 22.  The cohort18 summary panel was titled "type: 47/50 vs
    SPECTYPE, 44/50 for the target bits" over a matrix whose axes are the DESI
    target bits and Opus 5, and whose diagonal is 47.  The 44 is a THIRD
    quantity -- the target bits against Redrock's SPECTYPE -- on an axis the
    figure never draws.  A reader counting the cells cannot reach 44, and asked
    where it came from.

    So: any "a / n" in a matrix caption, where n is the matrix total, must have
    a equal to the diagonal.  A number from an undrawn axis has to be labelled
    somewhere other than the title of the matrix.
    """
    n = int(M.sum()) if hasattr(M, 'sum') else sum(map(sum, M))
    diag = int(sum(M[i][i] for i in range(len(M)))) if not hasattr(M, 'trace') \
        else int(M.trace())
    bad = [(int(a), int(b)) for a, b in
           re.findall(r'(\d+)\s*/\s*(\d+)', title)
           if int(b) == n and int(a) != diag]
    if bad:
        raise SystemExit(
            '%s: the caption quotes %s over a matrix whose diagonal is %d out '
            'of %d. A score from an axis the matrix does not draw cannot go in '
            'its title -- name that axis separately.'
            % (where, ', '.join('%d/%d' % t for t in bad), diag, n))
    return title


def check_file(path):
    """Scan a slide-generator source for banned labels.  Returns a count."""
    src = open(path, encoding='utf-8').read()
    hits = [b for b in BANNED if b.lower() in src.lower()]
    if hits:
        raise SystemExit('%s contains %s' % (path, ', '.join(map(repr, hits))))
    return 0


if __name__ == '__main__':
    n = 0
    for p in sys.argv[1:]:
        check_file(p); n += 1
        print('  ok    %s' % p)
    print('%d file(s) clean' % n)
