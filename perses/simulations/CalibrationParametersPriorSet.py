known_gamma_A_mag = (np.ones_like(frequencies) * 0.5)
known_gamma_A_phase = np.linspace(0, 180, len(frequencies))
known_gamma_rec_mag = (np.ones_like(frequencies) * 0.5)
known_gamma_rec_phase = np.zeros_like(frequencies)
known_gain = np.ones_like(frequencies)
known_offset = np.zeros_like(frequencies)
true_gamma_A_mag = known_gamma_A_mag +\
    (0.00001 * np.sin(2 * np.pi * frequencies / 20))
true_gamma_A_phase = known_gamma_A_phase + 0.01
true_gamma_rec_mag = known_gamma_rec_mag -\
    (0.00005 * np.sin(2 * np.pi * frequencies / 30))
true_gamma_rec_phase = known_gamma_rec_phase + 0.01
true_gain = known_gain +\
    (0.00001 * np.sin(2 * np.pi * frequencies / 80.))
true_offset = known_offset + 0.03

known_gamma_A =\
    known_gamma_A_mag * np.exp(1.j * np.radians(known_gamma_A_phase))
known_gamma_rec =\
    known_gamma_rec_mag * np.exp(1.j * np.radians(known_gamma_rec_phase))
true_gamma_A =\
    true_gamma_A_mag * np.exp(1.j * np.radians(true_gamma_A_phase))
true_gamma_rec =\
    true_gamma_rec_mag * np.exp(1.j * np.radians(true_gamma_rec_phase))

class CalibrationParametersPriorSet(object):
    def __init__(self, frequencies):
        self.true_gamma_A = 
        self.true_gamma_rec
        self.true_gain
        self.true_offset
    
    def draw(self):
        return\
        {\
            'gamma_A': self.draw_gamma_A(),\
            'gamma_rec': self.draw_gamma_rec(),\
            'gain': self.draw_gain(),\
            'offset': self.draw_offset()\
        }
    
    def draw_gamma_A(self):
        return self.true_gamma_A
    
    def draw_gamma_rec(self):
        return self.true_gamma_rec
    
    def draw_gain(self):
        return self.true_gain
    
    def draw_offset(self):
        return self.true_offset

