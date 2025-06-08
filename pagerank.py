import os
import random
import re
import sys
sys.argv = ["heredity.py", "corpus0"]
#sys.argv = ["heredity.py", "corpus1"]
#sys.argv = ["heredity.py", "corpus2"]
DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    links = corpus[page]
    num_links = len(links)
    transition_probs = {}
    for link in links:
        transition_probs[link] = damping_factor / num_links
    for page in corpus:
        if page not in transition_probs:
            transition_probs[page] = (1 - damping_factor) / len(corpus)
    return transition_probs

def sample_pagerank(corpus, damping_factor, n):
    page_ranks = {page: 0 for page in corpus}
    current_page = random.choice(list(corpus.keys()))
    for _ in range(n):
        transition_probs = transition_model(corpus, current_page, damping_factor)
        current_page = random.choices(list(transition_probs.keys()), weights=transition_probs.values())[0]
        page_ranks[current_page] += 1
    for page in page_ranks:
        page_ranks[page] /= n
    return page_ranks

def iterate_pagerank(corpus, damping_factor):
    page_ranks = {page: 1 / len(corpus) for page in corpus}
    while True:
        new_page_ranks = {}
        for page in corpus:
            links = corpus[page]
            num_links = len(links)
            rank = 0
            for link in links:
                rank += page_ranks[link] / num_links
            rank *= damping_factor
            rank += (1 - damping_factor) / len(corpus)
            new_page_ranks[page] = rank
        if all(abs(new_page_ranks[page] - page_ranks[page]) < 1e-8 for page in corpus):
            break
        page_ranks = new_page_ranks
    return page_ranks


if __name__ == "__main__":
    main()
