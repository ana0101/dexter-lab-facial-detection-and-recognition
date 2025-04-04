class Parameters:
    def __init__(self):
        # set the parameters
        self.characters = ['dad', 'deedee', 'dexter', 'mom', 'unknown']
        self.known_characters = ['dad', 'deedee', 'dexter', 'mom']
        self.dim_window = (36, 36) 
        self.dim_window_dad = (42, 36)
        self.dim_window_deedee = (36, 54)
        self.dim_window_dexter = (36, 42)
        self.dim_window_mom = (36, 36)
        self.img_resizes = [1, 0.75, 0.5, 0.25, 0.125, 0.1]
        self.dim_hog_cell = 6 
        self.cells_per_block = 2 
        self.orientations = 9  
        self.stride = 6  # cnn sliding window stride
        self.overlap = 0.3
        self.number_positive_examples = 4244 
        self.number_negative_examples = 16000 
        self.threshold = 0
        self.threshold_cnn = 0.5
        self.use_flip = True
