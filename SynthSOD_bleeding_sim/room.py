import json
import torch
import pathlib
import hashlib
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
from coords import sph2cart, cart2sph

class ExtendedRoom(pra.ShoeBox):
    def __init__(
        self,
        room_name,
        rt60,
        room_dims,
        fs,
        instrument_layout,
        channel_mapper,
        decca_rel_center = [.8,.5],
        decca_height = 3,
        close_mic_pattern_coeff = .25,
        air_absorption=True,
    ):
        e_absorption,max_order = pra.inverse_sabine(rt60,room_dims)
        materials = pra.Material(e_absorption)
        super().__init__(
            room_dims,
            fs=fs,
            materials=materials,
            air_absorption=air_absorption,
            max_order=max_order,
        )
        self.room_name = room_name
        self.rt60 = rt60
        self.room_dims = room_dims
        self.fs = fs
        self.decca_rel_center = decca_rel_center
        self.decca_height = decca_height
        self.instrument_layout = instrument_layout
        self.channel_mapper = channel_mapper
        self.close_mic_pattern_coeff = close_mic_pattern_coeff
        self.air_absoption_bool = air_absorption
        self.signature = self._compute_room_signature()

    def _compute_room_signature(self):
        params = [
            'room_name',
            'rt60',
            'room_dims',
            'fs',
            'decca_rel_center',
            'decca_height',
            'instrument_layout',
            'channel_mapper',
            'close_mic_pattern_coeff',
            'air_absorption_bool'
        ]
        param_dict = {param:self.__dict__.get(param) for param in params}
        param_hash = hashlib.md5(
            json.dumps(param_dict).encode()
        ).hexdigest()

        return f'{self.room_name}_{param_hash}'

    def set_layout(self):
        # set sound source positions
        sources_cart_coords = self._compute_sources_positions()
        for n in range(sources_cart_coords.shape[0]):
            self.add_source(sources_cart_coords[n,:])
        # set mic positions
        mic_cart_coords, mic_dir_patterns = self._compute_mic_setup(
            sources_cart_coords
        )
        print(f'Mic cart coords:\n{mic_cart_coords}')
        self.add_microphone_array(
            mic_cart_coords.T,  # PRA expects mic coords to have shape (coords,n_mics)
            directivity=mic_dir_patterns
        )

    def subplots(self):
        '''
        Representa la sala con las fuentes y micrófonos en 3d y
        vista desde arriba en dos subplots
        '''
        room_dim = self.shoebox_dim
        fig = plt.figure(figsize=(9,16),dpi=200)
        ax = [
            fig.add_subplot(1,2,1,projection='3d'),
            fig.add_subplot(1,1,1,projection='3d')
        ]
        # primer subplot de sala en 3d
        # self.plot(ax=ax[0])
        # ax[0].set_xlim([-1,room_dim[0]])
        # ax[0].set_ylim([-1,room_dim[1]])
        # ax[0].set_zlim([-1,room_dim[2]])
        # ax[0].set_xlabel('x (m)')
        # ax[0].set_ylabel('y (m)')
        # ax[0].set_zlabel('z (m)')
        # ax[0].set_title('3D view')
        # segundo subplot con vista desde arriba
        self.plot(ax=ax[1])
        ax[1].set_xlim([-1,room_dim[0]])
        ax[1].set_ylim([-1,room_dim[1]])
        ax[1].set_zlim([-1,room_dim[2]])
        ax[1].view_init(elev=90, azim=180)
        ax[1].set_zticks([])
        ax[1].set_title('Top-down view')
        ax[1].set_xlabel('x (m)')
        ax[1].set_ylabel('y (m)')
        ax[1].set_zlabel(' ')

        return fig,ax

    def rir_stack(self):
        # compute the RIRs (if not already done)
        if self.rir is None:
            self.compute_rir()
        # Max RIR length is fixed to the rt60 de la sala
        N = np.floor(self.rt60_theory()*self.fs).astype(int)
        RIR = np.zeros([self.n_sources,self.n_mics,N])
        for src_idx in range(self.n_sources):
            for mic_idx in range(self.n_mics):
                # rir is a list of lists so that the outer list is on
                # microphones and the inner list over sources
                rir = self.rir[mic_idx][src_idx]  
                if len(rir) <= N:
                    RIR[src_idx,mic_idx,0:len(rir)] = rir
                else:
                    RIR[src_idx,mic_idx,:] = rir[0:N]
        
        return RIR

    def save_rirs(self,parent_dir):
        RIR = self.rir_stack()
        # save RIR stack
        parent_dir = parent_dir / self.signature
        plots_dir = parent_dir / 'plots'
        plots_dir.mkdir(parents=True,exist_ok=True)
        np.save(parent_dir / 'RIR.npy',RIR)
        # save figures representing RIR measured at each mic for each source
        for ins_idx,ins_name in enumerate(self.instrument_layout.keys()):
            fig,ax = plt.subplots(4,4,figsize=(10,8),dpi=200,sharex=True,sharey=True)
            ax = ax.flatten()
            for mic_idx in range(self.n_mics):
                ax[mic_idx].plot(RIR[ins_idx,mic_idx,:],linewidth=.4)
                ax[mic_idx].set_title(f'RIR[{ins_idx},{mic_idx},:]')
            fig.suptitle(f'{ins_name} RIR measured at each mic')
            fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leaves top 5% for suptitle
            fig.savefig(plots_dir / f'{ins_name}.png')
            plt.close(fig)
    
    def _compute_sources_positions(self):
        # get decca coords
        decca_cart_coords = np.hstack([[
            dim*decca_rel_pos for dim,decca_rel_pos 
            in zip(self.room_dims,self.decca_rel_center)],
            self.decca_height])
        # compute each sound source cartesian coordinates with
        # regard to the local center (decca mic)
        sources_rel_sph_coords = np.stack(list(self.instrument_layout.values()))
        sources_cart_coords = np.stack(
            sph2cart(
            sources_rel_sph_coords[:,0],
            sources_rel_sph_coords[:,1],
            sources_rel_sph_coords[:,2],
            degrees=True
            )
        ).T + decca_cart_coords

        return sources_cart_coords

    def _compute_mic_setup(self,sources_cart_coords):
        mic_cart_coords = []
        mic_dir_patterns = []
        source_names = list(self.instrument_layout.keys())
        # Each sound source position serves as relative center
        # to compute its associate mic position and directive pattern
        mic_rel_cart_coords = np.array(sph2cart(
            1,      # 1 meter apart from the sound source
            90,     # same height
            180,    # closer to the decca central point
            degrees=True
        ))
        for src,mic_idx in self.channel_mapper.items():
            if src in ['Piccolo','coranglais']:
                continue
            # get the source index
            src_idx = source_names.index(src)
            # compute mic global cartesian coords
            mic_cart_coords.append(
                mic_rel_cart_coords + sources_cart_coords[src_idx,:]
            )
            # define mic polar pattern to point towards the sound source
            mic_dir_patterns.append(
                pra.CardioidFamily(
                    orientation=pra.DirectionVector(
                        azimuth=0,
                        colatitude=90,
                        degrees=True),
                    p=self.close_mic_pattern_coeff,
                )
            )
        
        # return each mic coordinates (numpy array) and their patterns
        return np.vstack(mic_cart_coords),mic_dir_patterns