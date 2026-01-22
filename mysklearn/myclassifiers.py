from mysklearn import myutils
from mysklearn.myutils import tdidt, tdidt_predict


class MyDecisionTreeClassifier:
    """
        Purpose: Represents a decision tree classifier.

        Attributes:
            X_train (list of list of obj): - The list of training instances (samples).
                                           - has shape: (n_train_samples, n_features)
            y_train (list of obj): - The target y values (labels corresponding to X_train).
                                   - has shape: y_train is n_samples
            tree (nested list): The extracted tree model.

        Notes:
            Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
            Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """
            Purpose: Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None


    def fit(self, X_train, y_train):
        """
            Purpose: Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

            Args:
                X_train (list of list of obj): - The list of training instances (samples).
                                               - has shape: (n_train_samples, n_features)
                y_train (list of obj): - The target y values (labels corresponding to X_train)
                                       - has shape: n_train_samples

            Notes:
                - Since TDIDT is an eager learning algorithm, this method builds a decision tree model from the training data.
                - Build a decision tree using the nested list representation described in class.
                - On a majority vote tie, choose first attribute value based on attribute domain ordering.
                - Store the tree in the tree attribute.
                - Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        
            - the goal of fit is to create the splits (the nested list) and decide probabilites at each leaf node.
        """
        self.X_train = X_train
        self.y_train = y_train
        
        # initialize a tree 
        tree = []

        # find number of features in list (by looking at the length of the first row in X_train).
        num_attributes = len(X_train[0])

        # get all unique attributes and sort them alphabetically.
        header = [f"att{i}" for i in range(num_attributes)]

        # stich together X_train and y_train
        train_data = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        
        # build attribute domains (the set of all values that an attribut can take)
        attribute_domains = {} # initialize a dict to store each attribute's unique values (i.e., att1="job_status" has values 1, 2, 3)
        
        # iterate through each col (each attribute corresponds to 1 col) and find unique values in that col using list(set())
        for idx in range(num_attributes):
            attribute_domains[header[idx]] = sorted(list(set(row[idx] for row in X_train)))
        
        # make a copy a header, b/c python is pass by object reference and tdidt will be removing attributes from available_attributes
        available_attributes = header.copy()
        
        # create the decision tree using top down induction decision tree (tdidt)
        # self.tree = tdidt(train_data, available_attributes)
        self.tree = tdidt(train_data, available_attributes, header, attribute_domains)

        

    def predict(self, X_test):
        """
            Purpose: Makes predictions for test instances in X_test.

        Args:
            X_test (list of list of obj): - The list of testing samples
                                          - has shape: (n_test_samples, n_features)

        Returns:
            y_predicted (list of obj): The predicted target y values (labels corresponding to X_test)
        """
        # find number of features in list.
        num_attributes = len(X_test[0])

        # get all unique attributes and sort them alphabetically.
        header = [f"att{i}" for i in range(num_attributes)]

        y_predicted = []

        # generate a prediction for each instance in the test set (the instances we want to classify)
        for instance in X_test:
            pred = tdidt_predict(self.tree, instance, header)
            y_predicted.append(pred)

        return y_predicted 


    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """
            Purpose: Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

            Args:
                attribute_names (list of str or None): - A list of attribute names to use in the decision rules
                                                    - if None (the list was not provided), use the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
                class_name(str): - A string to use for the class name in the decision rules
                                ("class" if a string is not provided and the default name "class" should be used).
            
            Note: - Leaf subtree lists are stored as [type_of_node (e.g., "Attribute", "Value", "Leaf"), node_label (e.g., "True", "False"), numerator_probability, denominator_probability]
                  - Attribute subtree lists are stored as [type_of_node, attribute]
                  - I did reference ChatGPT to guide me through how I would approach this function because I was getting a bit lost in the logic due to the recursion element here.
        """
        # get the number of attributes in the dataset. 
        num_attributes = len(self.X_train[0])

        # handle attribute_names=None parameter.
        if attribute_names is None:
            attribute_names = [f"att{i}" for i in range(num_attributes)]
        
        # traverse tree until one of the following conditions have been met.
        def recurse(subtree, conditions):
            curr_node = subtree[0] # grab the current node type. (e.g., "Attribute", "Value", "Leaf"). 
            
            # base case 1: if leaf, print decision rule:
            if curr_node == "Leaf":
                label = subtree[1] # get the label for the current node. (e.g., "True", "False").
                rule_str = " AND ".join(conditions) # grab all conditions that have led us to this leaf node.
                print(f"IF {rule_str} THEN {class_name} = {label}") # print the rule.

            # base case 2: if attribute node, recurse/iterate down through each branch and "append" a rule as we go.
            elif curr_node == "Attribute":
                att_name = subtree[1] # get the attribute that the current node splits on. 

                for branch in subtree[2:]: # iterate over each branch off of the current node.
                    att_val = branch[1] # get the value of the attribute for the current branch
                    val_subtree = branch[2] # get the subtree that goes off of the current branch.
                    new_rule = conditions + [f"{att_name} == {att_val}"] # add this rule to the decision rule string.
                    recurse(val_subtree, new_rule) # recursively continue to go through the subtree.

        recurse(self.tree, [])

        


    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this





