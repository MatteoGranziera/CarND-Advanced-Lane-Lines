import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, max_saved_iterations = 5):
        self.max_saved_iterations = max_saved_iterations
        self.all_losted_iterations = 0
        self.losted_iterations = max_saved_iterations
        self.need_reset = True
        
        self.reset()
        
        # Based on US specifications
        self.ym_per_pix = 44/100 # meters per pixel in y dimension
        
        # Accepted coefficients diffs between lines
        self.margin_a = 0.0002
        self.margin_b = 0.2
        self.margin_c = 60
        
    def reset(self):
        self.losted_iterations = self.max_saved_iterations
         # was the line detected in the last iteration?
        self.detected = False 
        
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        
        # average x values of the fitted line over the last n iterations
        self.bestx = None     
        
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        
        #x values for detected line pixels
        self.allx = None  
        
        #y values for detected line pixels
        self.ally = None 
        
    def measure_curvature(self):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = self.ym_per_pix
        
        fit_cr = self.current_fit
        ploty = self.ally

        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        # Calculate radius of curvature
        curverad = (1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2 )**(3/2) / np.absolute(2*fit_cr[0]) 

        self.radius_of_curvature = int(curverad)

    def sanity_check(self):
        valid = False
        
        # Recalculate coefficients diff 
        if self.best_fit != None:
            a, b, c = np.absolute(self.best_fit - self.current_fit)
        
            valid = (
                (-self.margin_a <= a <= self.margin_a) & 
                (-self.margin_b <= b <= self.margin_b) &
                (-self.margin_c <= c <= self.margin_c) )
        else: 
            valid = True        
        
        # Update iteration
        if valid == True:
            self.losted_iterations = 0
        else:
            self.losted_iterations = self.losted_iterations + 1
            self.all_losted_iterations = self.all_losted_iterations + 1
            
        if self.losted_iterations >= self.max_saved_iterations:
            self.need_reset = True
        
        # Update last check flag
        self.detected = valid
        
    def get_line(self, pix_y, pix_x, height):
        
        # Get second order polynomial from new data
        fit = np.polyfit(pix_y, pix_x, 2)

        # Check if last n iterations the line was not detected correctly
        if self.need_reset == True:
            self.reset()
            
        # Save current fit
        self.current_fit = fit
        
        # Make sanity checks
        self.sanity_check()
        
#         print("Detected: ", self.detected)
        
        # If the line passed sanity checks
        if self.detected == True:

            # If the new line has passed the sanity check 
            # I'll give more importance than the past averages
            if self.need_reset == True:
                self.best_fit = fit
                self.need_reset = False
            else:
                self.best_fit = (self.best_fit + fit) / 2
            
            new_fit = self.best_fit
        
            # Generate x and y values for plotting
            fity = np.linspace(0, height-1, height )
            fitx = new_fit[0]*fity**2 + new_fit[1]*fity + new_fit[2]

            # Save latest data
            self.allx = fitx
            self.ally = fity
        
            # Append latest line pixels
            self.recent_xfitted.append(fitx)

        # Remove old data
        self.remove_old_iteration()
        self.measure_curvature()

#         print(self.radius_of_curvature)
        
        return self.ally,  self.allx
    
    def calculat_best_fit(self):
        iterations = len(self.recent_xfitted)
        self.bestx = np.mean(self.recent_xfitted, axis=0)
    
    def remove_old_iteration(self):
        if(len(self.recent_xfitted) >= self.max_saved_iterations):
            self.recent_xfitted = self.recent_xfitted[1:]
            self.recent_xfitted = self.recent_xfitted[1:]
            
        self.calculat_best_fit()
            
