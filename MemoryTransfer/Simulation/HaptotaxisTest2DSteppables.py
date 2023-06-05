from cc3d.cpp.PlayerPython import * 
from cc3d import CompuCellSetup
from cc3d.core.PySteppables import *
import random 
from math import *
import numpy as np
from random import uniform


#global variables#

# mu_0={{mu_0}}
# k_al={{k_al}}
# prop=0.01
# pix_to_um = 4.0
# correlation_L = 100.0
f0 = {{f0}}
f_star = {{f_star}}
# f_pore_star = {{f_pore_star}}
n_f = {{n_f}}
phi_0 = {{phi_0}}
x_dim = 300
y_dim = 75
col_start_loc = 150
n_gamma = {{n_gamma}}
# n_alpha = {{n_alpha}}
gamma0 = {{gamma0}}
r_phi_0 = {{r_phi_0}}
r_psi = {{r_psi}}
n_x = {{n_x}}
half_max_psi = {{half_max_psi}}

speed_scale = 10


class PhiSolverSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        

    def start(self):
        global f0
        global gamma0
        global r_phi_0
        global r_phi_b_0
        global r_psi
        global n_f
        global n_gamma
        global phi_0
        global f_star
        # global f_pore_star
        
        Phi_model_string = """model PhiModel()

        f0 = 2.3
        phi_0 = 1       
        r_phi_0 = 0.01
        r_psi = 0.000
        n_f = 2
        n_gamma = 4
        f_star = 2.5
        f_pore_star = 1.5
        ecm_switch = 0
        
        f_bar := f0/f_star
        f_pore:= f0/f_pore_star
        phi_f := (f_bar^n_f)/(1+f_bar^n_f);
        phi_p := (f_pore^4)/(1+f_pore^4);

        -> phi; r_phi_0*ecm_switch*(phi_f - phi)
        -> psi; r_psi*ecm_switch*(phi_f - psi)

        phi = phi_0
        psi = phi_0
        
        end"""

        options = {'stiff': True}
        # self.set_sbml_global_options(options)
        step_size = 1

        for cell in self.cell_list:
            self.add_antimony_to_cell(model_string=Phi_model_string,
                                      model_name='phiModel',
                                      cell=cell,
                                      step_size=step_size)

        for cell in self.cell_list:
            cell.sbml.phiModel['f0'] = f0
            cell.sbml.phiModel['f_star'] = f_star           
            # cell.sbml.phiModel['f_pore_star'] = f_star           
            cell.sbml.phiModel['r_phi_0'] = r_phi_0    
            cell.sbml.phiModel['r_psi'] = r_psi    
            cell.sbml.phiModel['n_f'] = n_f
            cell.sbml.phiModel['phi'] = phi_0
            cell.sbml.phiModel['psi'] = phi_0

        # for cell in self.cell_list:
            # dist = (140 - cell.xCOM)
            # if dist > 0.0:
                # #cell.sbml.phiModel['phi'] = -(phi_0/135.0)*dist + phi_0*137.5/135 #phi0 NEAR FRONT AND 0.0 AT THE BACK
                # cell.sbml.phiModel['phi'] = -(phi_0/135.1)*dist + phi_0*137.6/135 

    def step(self, mcs):
        global half_max_psi
        #print("98")
        fiber_field = self.field.fiber
        for cell in self.cell_list:
            # alpha_cell = alpha_field[cell.xCOM, cell.yCOM, 0]
            # cell.sbml.phiModel['alpha'] = alpha_cell
            fiber_under_cell = fiber_field[cell.xCOM, cell.yCOM, 0]
            cell.sbml.phiModel['ecm_switch'] = (fiber_under_cell != 0)
            cell.sbml.phiModel['r_psi'] = (r_phi_0*mcs**4)/(half_max_psi**4 + mcs**4)
            
            # if mcs == 800:#CONTRACTILITY INHIBITION
                # cell.sbml.phiModel['phi'] = 0.1
                # cell.sbml.phiModel['psi'] = 0.1
                # cell.sbml.phiModel['r_phi_0'] = 0.0
                # cell.sbml.phiModel['r_psi'] = 0.0

            # if mcs == 1200:# LOSS OF MEMORY
                # cell.sbml.phiModel['r_psi'] = 0.1    

                
        self.timestep_sbml()

    def finish(self):
        # this function may be called at the end of simulation - used very infrequently though
        return


class AssignMotilitySteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
    
    def start(self):

        self.shared_steppable_vars['current_LE_x'] = 0.0
        self.shared_steppable_vars['current_alpha_total_norm'] = 0.0

        self.plot_win1 = self.add_new_plot_window(
            title='Average speed',
            x_axis_title='MonteCarlo Step (MCS)',
            y_axis_title='LE speed',
            x_scale_type='linear',
            y_scale_type='linear',
            grid=True # only in 3.7.6 or higher
        )
        self.plot_win1.add_plot("AverageLeSpeed", style='Dots', color='red', size=5)                

        self.plot_win2 = self.add_new_plot_window(
            title='Average phi',
            x_axis_title='MonteCarlo Step (MCS)',
            y_axis_title='phi',
            x_scale_type='linear',
            y_scale_type='linear',
            grid=True # only in 3.7.6 or higher
        )
        self.plot_win2.add_plot("Avg_LE_phi", style='Dots', color='red', size=5)                

        self.plot_win4 = self.add_new_plot_window(
            title='Average psi',
            x_axis_title='MonteCarlo Step (MCS)',
            y_axis_title='psi',
            x_scale_type='linear',
            y_scale_type='linear',
            grid=True # only in 3.7.6 or higher
        )
        self.plot_win4.add_plot("Avg_LE_psi", style='Dots', color='red', size=5)                

        self.plot_win3 = self.add_new_plot_window(
            title='total alpha',
            x_axis_title='MonteCarlo Step (MCS)',
            y_axis_title='Variables',
            x_scale_type='linear',
            y_scale_type='linear',
            grid=True # only in 3.7.6 or higher
        )
        self.plot_win3.add_plot("total_alpha", style='Dots', color='red', size=5)                


        self.plot_win5 = self.add_new_plot_window(
            title='nx actual',
            x_axis_title='MonteCarlo Step (MCS)',
            y_axis_title='Variables',
            x_scale_type='linear',
            y_scale_type='linear',
            grid=True # only in 3.7.6 or higher
        )
        self.plot_win5.add_plot("n_x_actual", style='Dots', color='red', size=5)                 

        self.plot_win6 = self.add_new_plot_window(
            title='avg_LE_mu',
            x_axis_title='MonteCarlo Step (MCS)',
            y_axis_title='Variables',
            x_scale_type='linear',
            y_scale_type='linear',
            grid=True # only in 3.7.6 or higher
        )
        self.plot_win6.add_plot("avg_LE_mu", style='Dots', color='red', size=5)                 

        self.plot_win7 = self.add_new_plot_window(
            title='avg_LE_gam',
            x_axis_title='MonteCarlo Step (MCS)',
            y_axis_title='Variables',
            x_scale_type='linear',
            y_scale_type='linear',
            grid=True # only in 3.7.6 or higher
        )
        self.plot_win7.add_plot("avg_LE_gam", style='Dots', color='red', size=5)                 


        
        for cell in self.cell_list:
            dist = abs(cell.xCOM - (col_start_loc+5) )
            if dist <= 15.0:
                cell.dict['rm_abi'] = 1
            else:
                cell.dict['rm_abi'] = 0
        
        
    def step(self, mcs):
        #print("177")
        pop_com_x = 0
        for cell in self.cell_list:
            pop_com_x += cell.xCOM
        pop_com_x = pop_com_x/len(self.cell_list)
        
        global gamma0
        global f0
        global n_gamma
        # global n_alpha
        # global x_star
        global n_x
        global f_star
        
        # fiber = self.field.fiber
        field_fiber = self.field.fiber
        alpha_field = self.field.alpha
        phi_field = self.field.phi_field
        psi_field = self.field.psi_field
        gam_field = self.field.gam_field
        mu_field = self.field.mu_field
        rm_abi_field = self.field.rm_abi
        phi_field[:,:,:] = 0.0
        psi_field[:,:,:] = 0.0
        gam_field[:,:,:] = 0.0
        mu_field[:,:,:] = 0.0
        rm_abi_field[:,:,:] = 0.0
        
        
        count_leaders = 0
        LE_x = 0
        mu_LE = 0
        avg_LE_alpha = 0.0
        avg_phi = 0.0
        avg_psi = 0.0
        phi_counter = 0
        alpha_total = 0.0
        avg_LE_mu = 0.0
        avg_LE_gam = 0.0
        
        for cell in self.cell_list_by_type(self.CELL):
            is_boundary_cell = 0
            fbn = 0.0
            mu_total = 0.0
            for neighbor, common_surface_area in self.get_cell_neighbor_data_list(cell):
                if not neighbor and cell.xCOM > pop_com_x:
                    is_boundary_cell = 1
                    fbn = common_surface_area/cell.surface #fbn =  free boundary normalized
                    LE_x += cell.xCOM
                    count_leaders += 1
                    avg_LE_alpha += alpha_field[cell.xCOM,cell.yCOM,0]
                    # cell.type = self.LEADER
            if is_boundary_cell == 0:
                fbn = 0.0

            if(cell.dict['rm_abi'] == 1):
                avg_phi += cell.sbml.phiModel['phi']
                avg_psi += cell.sbml.phiModel['psi']
                # avg_LE_mu += mu_cell*(f0_cell !=0 )
                phi_counter +=1
        
        avg_LE_alpha /= count_leaders
        LE_x = LE_x/count_leaders
        LE_mu_counter = 0
        f_bar = f0/f_star

        current_alp_tot_norm = self.shared_steppable_vars['current_alpha_total_norm']
        # n_x_actual = n_x/( 1+(f_bar**4)*current_alp_tot_norm) 
        n_x_actual = n_x #- (f_bar)*current_alp_tot_norm
        # n_x_actual = (n_x_actual < 0.05)*(0.05-n_x_actual) + n_x_actual
        for x_running in range(col_start_loc, x_dim ):
            for y_running in range(y_dim):
                # #print(x_running,y_running)
                # #print(self.cell_field[x_running,y_running,0])  
                
                if self.cell_field[x_running,y_running,0] == None:
                # if 1==2:
                    # alpha_field[x_running,y_running,0] = avg_LE_alpha*(x_running-LE_x<0) + (x_running-LE_x>=0)*avg_LE_alpha*(1/(1 + ((x_running-LE_x)/x_star)**4 ) ) #1/(1+R^4) FORM
                    # #print("258")
                    alpha_running = alpha_field[x_running,y_running,0]
                    
                    if x_running <= LE_x:
                        alpha_field[x_running,y_running,0] = (avg_LE_alpha - alpha_running)*(avg_LE_alpha>=alpha_running) + alpha_running
                        # #print("261")
                    else:
                        # n_x_actual = n_x/( 1+(f_bar**4)*current_alp_tot_norm) 
                        # n_x_actual = n_x
                        alpha_propagation = avg_LE_alpha/(x_running-LE_x)**(n_x_actual) 
                        # print("ap",alpha_propagation)
                        # print("nx",n_x/( (f0**2)*(1+ avg_LE_alpha) ) )
                        alpha_field[x_running,y_running,0] = ( (avg_LE_alpha - alpha_running)*(avg_LE_alpha>=alpha_running) + alpha_running )*(x_running-LE_x<1) + (x_running-LE_x>=1)*( (alpha_propagation - alpha_running)*( alpha_propagation >= alpha_running) + alpha_running )     # 1/R FORM
                        # #print("264")
                alpha_total += alpha_field[x_running,y_running,0]        

        alpha_total_norm = (alpha_total/y_dim)/(x_dim-col_start_loc)
        self.shared_steppable_vars['current_alpha_total_norm'] = alpha_total_norm
        
        LE_mu_counter = 0
        for cell in self.cell_list_by_type(self.CELL):
            #f0_cell = field_fiber[cell.xCOM,cell.yCOM,0]#maybe change this to f0*(left of col_start_loc or not)
            f0_cell = f0 #maybe change this to f0*(left of col_start_loc or not)
            phi_cell = cell.sbml.phiModel['phi']
            psi_cell = cell.sbml.phiModel['psi']
            # gamma_cell = gamma0*(f0**n_gamma)-(alpha_field[cell.xCOM,cell.yCOM,0])**(n_alpha)  
            # gamma_cell = gamma0*(f0_cell**n_gamma)-(alpha_field[cell.xCOM,cell.yCOM,0])**(n_alpha)  
            # gamma_cell = gamma0*(f0_cell**n_gamma) - (f_bar**n_alpha)*(alpha_total_norm)  
            gamma_cell = gamma0*(f0_cell**n_gamma) - f_bar*(alpha_total_norm)  
            gamma_cell = gamma_cell*(gamma_cell>0)
            mu_cell = (phi_cell - gamma_cell)
            
            if mu_cell <= 0:
                mu_cell = 0.05 #assigning a very smal value of mu, so that they dont retract back 

            pixel_list = self.get_cell_pixel_list(cell)
            for pixel_tracker_data in pixel_list:
                px = pixel_tracker_data.pixel
                phi_field[px.x, px.y, px.z] = phi_cell
                psi_field[px.x, px.y, px.z] = psi_cell
                gam_field[px.x, px.y, px.z] = gamma_cell
                mu_field[px.x,px.y,px.z] = mu_cell
                rm_abi_field[px.x,px.y,px.z] = cell.dict['rm_abi']
            
            cell.lambdaVecX = -mu_cell*speed_scale  # force component pointing along X axis - towards positive X's
            cell.lambdaVecY = 0.0  # force component pointing along Y axis - towards negative Y's
            cell.lambdaVecZ = 0.0  # force component pointing along Z axis

            if(cell.dict['rm_abi'] == 1 and cell.xCOM > col_start_loc): #only cells in collagen
                avg_LE_mu += mu_cell
                avg_LE_gam += gamma_cell
                LE_mu_counter +=1
        
        avg_LE_mu /= LE_mu_counter
        avg_LE_gam /= LE_mu_counter
#follower cells have same mu as leaders
        for cell in self.cell_list:
            if(cell.dict['rm_abi'] == 0 or cell.xCOM <= col_start_loc):
                cell.lambdaVecX = -avg_LE_mu*speed_scale  # force component pointing along X axis - towards positive X's
                pixel_list = self.get_cell_pixel_list(cell)
                for pixel_tracker_data in pixel_list:
                    px = pixel_tracker_data.pixel
                    mu_field[px.x,px.y,px.z] = avg_LE_mu

        # #print("266")
        if mcs == 0:        
            self.shared_steppable_vars['current_LE_x'] = LE_x

            self.plot_win2.add_data_point('Avg_LE_phi', mcs, (avg_phi)/phi_counter )
            self.plot_win4.add_data_point('Avg_LE_psi', mcs, (avg_psi)/phi_counter )
            self.plot_win3.add_data_point('total_alpha',mcs, alpha_total_norm)
            self.plot_win5.add_data_point('n_x_actual',mcs, n_x_actual)
            self.plot_win6.add_data_point('avg_LE_mu',mcs, avg_LE_mu)
            self.plot_win7.add_data_point('avg_LE_gam',mcs, avg_LE_gam)            
            #print("269")
        elif mcs%100 == 0:        
            delta_LE_x = LE_x - self.shared_steppable_vars['current_LE_x'] # NOT INSTANTANEOUS, BUT AVG
            # self.shared_steppable_vars['current_LE_x'] = LE_x
            
            self.plot_win1.add_data_point('AverageLeSpeed', mcs, (delta_LE_x)/mcs)
            self.plot_win2.add_data_point('Avg_LE_phi', mcs, (avg_phi)/phi_counter )
            self.plot_win4.add_data_point('Avg_LE_psi', mcs, (avg_psi)/phi_counter )
            self.plot_win3.add_data_point('total_alpha',mcs, alpha_total_norm)
            self.plot_win5.add_data_point('n_x_actual',mcs, n_x_actual)
            self.plot_win6.add_data_point('avg_LE_mu',mcs, avg_LE_mu)
            self.plot_win7.add_data_point('avg_LE_gam',mcs, avg_LE_gam)

    def finish(self):
        if self.output_dir is not None:
            png_output_path = Path(self.output_dir).joinpath("Plot_LE_speed_f0_" + str(f0) +"_phi0_"+str(phi_0)+ ".png")
            self.plot_win1.save_plot_as_png(png_output_path)
            csv_output_path = Path(self.output_dir).joinpath("data_LE_speed_f0_" + str(f0) +"_phi0_"+str(phi_0)+ ".csv")
            self.plot_win1.save_plot_as_data(csv_output_path, CSV_FORMAT)

            png_output_path = Path(self.output_dir).joinpath("Plot_phi_f0_" + str(f0) +"_phi0_"+str(phi_0)+ ".png")
            self.plot_win2.save_plot_as_png(png_output_path)
            csv_output_path = Path(self.output_dir).joinpath("data_phi_f0_" + str(f0) +"_phi0_"+str(phi_0)+ ".csv")
            self.plot_win2.save_plot_as_data(csv_output_path, CSV_FORMAT)

            png_output_path = Path(self.output_dir).joinpath("Plot_psi_f0_" + str(f0) +"_phi0_"+str(phi_0)+ ".png")
            self.plot_win4.save_plot_as_png(png_output_path)
            csv_output_path = Path(self.output_dir).joinpath("data_psi_f0_" + str(f0) +"_phi0_"+str(phi_0)+ ".csv")
            self.plot_win4.save_plot_as_data(csv_output_path, CSV_FORMAT)

            png_output_path = Path(self.output_dir).joinpath("Plot_alpha_f0_" + str(f0) +"_phi0_"+str(phi_0)+ ".png")
            self.plot_win3.save_plot_as_png(png_output_path)
            csv_output_path = Path(self.output_dir).joinpath("data_alpha_f0_" + str(f0) +"_phi0_"+str(phi_0)+ ".csv")
            self.plot_win3.save_plot_as_data(csv_output_path, CSV_FORMAT)

            png_output_path = Path(self.output_dir).joinpath("Plot_nx_f0_" + str(f0) +"_phi0_"+str(phi_0)+ ".png")
            self.plot_win5.save_plot_as_png(png_output_path)
            csv_output_path = Path(self.output_dir).joinpath("data_nx_f0_" + str(f0) +"_phi0_"+str(phi_0)+ ".csv")
            self.plot_win5.save_plot_as_data(csv_output_path, CSV_FORMAT)

            png_output_path = Path(self.output_dir).joinpath("Plot_avgMu_f0_" + str(f0) +"_phi0_"+str(phi_0)+ ".png")
            self.plot_win6.save_plot_as_png(png_output_path)
            csv_output_path = Path(self.output_dir).joinpath("data_avgMu_f0_" + str(f0) +"_phi0_"+str(phi_0)+ ".csv")
            self.plot_win6.save_plot_as_data(csv_output_path, CSV_FORMAT)

            png_output_path = Path(self.output_dir).joinpath("Plot_avgGam_f0_" + str(f0) +"_phi0_"+str(phi_0)+ ".png")
            self.plot_win7.save_plot_as_png(png_output_path)
            csv_output_path = Path(self.output_dir).joinpath("data_avgGam_f0_" + str(f0) +"_phi0_"+str(phi_0)+ ".csv")
            self.plot_win7.save_plot_as_data(csv_output_path, CSV_FORMAT)



class FiberConcentrationCaseBRandom50(SteppableBasePy):
    def __init__(self,frequency=1):
        SteppableBasePy.__init__(self,frequency)
        
    def start(self):
        field_fiber      = self.field.fiber
        # alpha_field      = self.field.alpha
        # f0 = self.shared_steppable_vars['f0']
        global f0
        for x,y,z in self.every_pixel():   
            if x >= col_start_loc:
                field_fiber[x,y,z]  = f0
                # alpha_field[x,y,z]  = 1
            else:
                field_fiber[x,y,z]  = 0

class tracks_n_plots(SteppableBasePy):

    def __init__(self,frequency=100):

        SteppableBasePy.__init__(self,frequency)
        
    # def start(self):
        # """
        # any code in the start function runs before MCS=0
        # """
    def step(self,mcs):
        alpha_field = self.field.alpha
        output_dir = self.output_dir

        if output_dir is not None:
            output_path = Path(output_dir).joinpath('alpha_' + str(mcs) + '.dat')
            with open(output_path, 'w') as fout:

                for y_running in range(y_dim):
                    for x_running in range(col_start_loc, x_dim ):
                        fout.write('{}\n'.format(alpha_field[x_running, y_running, 0]))
                    # fout.write('\n')    

            
           

    # def finish(self):
        # """
        # Finish Function is called after the last MCS
        # """
        # output_dir = self.output_dir

        # if output_dir is not None:
            # output_path = Path(output_dir).joinpath('step_' + str(1).zfill(3) + '.dat')
            # with open(output_path, 'w') as fout:

                # for x in range(len(self.shared_steppable_vars['current_LE_x'])):
                    # fout.write('{}\n'.format(self.shared_steppable_vars['current_LE_x'][x])) 

        # for x in range(80):
            # self.shared_steppable_vars['speed_leaders'][x] = self.shared_steppable_vars['speed_leaders'][x+1] - self.shared_steppable_vars['speed_leaders'][x]
        # self.shared_steppable_vars['speed_leaders'] = self.shared_steppable_vars['speed_leaders']/25

        # if output_dir is not None:
            # output_path = Path(output_dir).joinpath('speed_' + str(1).zfill(3) + '.dat')
            # with open(output_path, 'w') as fout:

                # for x in range(80):
                    # fout.write('{}\n'.format(self.shared_steppable_vars['speed_leaders'][x])) 
        
