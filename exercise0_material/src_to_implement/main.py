from generator import ImageGenerator
import pattern
import numpy as np

checker_test = pattern.Checker(1024, 32)
checker_test.draw()
checker_test.show()

circle_test = pattern.Circle(1024, 200, (700, 400))
circle_test.draw()
circle_test.show()

spectrum_test = pattern.Spectrum(100)
spectrum_test.draw()
spectrum_test.show()

file_path = 'data\exercise_data'
label_path = 'data\Labels.json'
gen = ImageGenerator(file_path, label_path, 50, [32, 32, 3], rotation=True, mirroring=True, shuffle=False)
gen.show()