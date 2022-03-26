"""
main.py
    Main thread of execution

    @author Nicholas Nordstrom
"""
import data_loader
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

INPUT_FILE = "Emojisets_.xlsx"


def main():
    # Task 1 Define and Name Algorithms
    """
    Define and name at least one algorithm for each of the following terms: Frequent Itemset Mining,
    Association Rule Mining, Sequence, Element (Transaction), Event (Item), and Sequential Pattern.
    """

    # Task 2 Apply Algorithms
    """
    Given a set of tweets in the associated files (uploaded to Brightspace), extract the Emojisets and
    then apply the Frequent Itemset Mining, Association Rule Mining, and Sequential Pattern to them.
    """

    # read data
    df = data_loader.read_excel(INPUT_FILE)
    print('DataFrame Columns:\n', df.columns)
    print('DataFrame Values:\n', df)

    dataset = df['Emojiset'].replace(',', '', regex=True)  # .replace(' ', '', regex=True).replace('️', '', regex=True)
    print('Dataset:\n', dataset)

    #   Frequent Itemset Mining
    # --------------------------------------------------------------
    # preprocess
    te = TransactionEncoder()
    te_ary = te.fit_transform(dataset)
    df_preprocessed = data_loader.to_dataframe(te_ary, columns=te.columns_)

    print('Processed Dataframe Columns:\n', df_preprocessed.columns)
    print('Processed Dataframe Values:\n', df_preprocessed)

    # process
    result = data_loader.to_dataframe(apriori(df_preprocessed, min_support=.1, use_colnames=True))
    print('Result:\n', result.sort_values('support', ascending=False))

    # Task 3 Algorithm Analysis
    """
    Analysis the result from step 2 and argue which technique (Frequent Itemset Mining, Association
    Rule Mining, or Sequential Pattern) should be used with the Emojiset and why, while also stating
    why the other techniques should not be used.
    """

    # Task 4 Maximal and Closed Emojisets.
    """
    Find the maximal and closed Emojisets.
    """

    # Task 5 Write Report
    """
    Write a report about your argument, use correct terminology, give the algorithms, and present
    results.
    """


if __name__ == "__main__":
    main()
