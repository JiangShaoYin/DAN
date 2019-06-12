from lasagne.layers import Layer

class LandmarkInitLayer(Layer): # increment 增量
    def __init__(self, increments, init_landmarks, **kwargs):
        super(LandmarkInitLayer, self).__init__(increments, **kwargs)

        self.init_landmarks = init_landmarks.flatten() # flatten()将多维数据，变成1维数据

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs): # input的数据 + 初始的landmark，将其结果返回。
        output = input + self.init_landmarks
        return output
