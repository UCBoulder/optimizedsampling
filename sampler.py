class Sampler:
    '''
    Class for sampling
    '''
    def __init__(self, *datasets, rule="random", loc_emb=None):
        '''Initialize a new Sampler instance.

        Args:
            datasets: data to sample from (df or array)
            rule: use rule to sample ("random", "image", "satclip")
            loc_emb: satclip embeddings if rule is satclip
        '''
        i = 0
        for dataset in datasets:
            if isinstance(dataset, pd.DataFrame):
                dataset = dataset.to_numpy()
            setattr(self, f"dataset{i+1}", dataset)
            i += 1
        
        self.rule = rule

        if loc_emb is not None:
            self.loc_emb = loc_emb

    '''
    Determine indexes of subset to sample
    '''
    def subset_idxs(self, n=0):
        if self.rule=="random":
            subset_idxs = np.random.choice(len(self.dataset1), size=n, replace=False)

        if self.rule=="image":
            subset_idxs = sampling_with_scores(self.dataset1, n, v_optimal_design)

        if self.rule=="satclip":
            subset_idxs = sampling_with_scores(self.loc_emb, n, v_optimal_design)

        return subset_idxs
    
    def sample(self, n=0):
        subset_idxs = self.subset_idxs(n)

        i = 1
        while True:
            dataset = getattr(self, f"dataset{i}", None)
            if dataset is None:
                return
            yield dataset[subset_idxs]
            i += 1