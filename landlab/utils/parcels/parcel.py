import numpy as np
import scipy.constants

from landlab.data_record import DataRecord


_OUT_OF_NETWORK = -2


class SedimentPulser:

    """
    Parameters
    ----------
    grid : ModelGrid
        landlab *ModelGrid* to place sediment parcels on.
    mannings_n : float, optional
        Mannings's n.
    tau_critical : float, optional
        Critical shear stress.
    rho_sediment : float, optional
        Sediment grain density [kg / m^3].
    rho_water : float, optional
        Density of water [kg / m^3].
    gravity : float, optional
        Accelertion due to gravity [m / s^2].
    std_dev : float, optional
        Standard deviation of lognormal distribution of grain size.

    Examples
    --------
    >>> from landlab import NetworkModelGrid
    >>> from landlab.utils.parcels import SedimentPulser

    >>> y_of_node = (0, 100, 200, 200, 300, 400, 400, 125)
    >>> x_of_node = (0, 0, 100, -50, -100, 50, -150, -100)
    >>> nodes_at_link = ((1, 0), (2, 1), (1, 7), (3, 1), (3, 4), (4, 5), (4, 6))
    >>> grid = NetworkModelGrid((y_of_node, x_of_node), nodes_at_link)
    >>> grid.at_link["channel_width"] = np.full(grid.number_of_links, 1.0)  # m
    >>> grid.at_link["channel_slope"] = np.full(grid.number_of_links, .01)  # m / m
    >>> grid.at_link["reach_length"] = np.full(grid.number_of_links, 100.0)  # m

    >>> def time_to_pulse(time):
    ...     return True
    >>> make_pulse = SedimentPulser(grid, time_to_pulse=time_to_pulse)
    >>> d50 = 1.5
    >>> parcels = make_pulse(0.0, d50)
    """
    def __init__(
        self,
        grid,
        parcels=None,
        rho_sediment=2650.0,
        std_dev=2.1,
        time_to_pulse=None,
    ):
        self._grid = grid
        self._parcels = parcels
        self._rho_sediment = rho_sediment
        self._std_dev = std_dev

        if time_to_pulse is None:
            self._time_to_pulse = lambda time: True
        else:
            self._time_to_pulse = time_to_pulse
            

    def __call__(self, time, d50, n_parcels_at_link=100, links=[0,3,4], max_parcel_volume = 0.05):
        if not self._time_to_pulse(time):
            return self._parcels

        links = np.asarray(links).reshape(-1)
        n_parcels_at_link, _ = np.broadcast_arrays(n_parcels_at_link, links)
        d50, _ = np.broadcast_arrays(d50, links)

        d84 = d50 * self._std_dev

        total_parcel_volume_at_link = calc_total_parcel_volume(
            self._grid.at_link["channel_width"][links],
            self._grid.at_link["reach_length"][links],
            d84 * 2.0 * 2.0,
        )

        variables, items = _pulse_characteristics(
            time,
            links,
            n_parcels_at_link,
            total_parcel_volume_at_link,
            max_parcel_volume,
            d50,
            self._std_dev,
            self._rho_sediment,
            0.0,
        )

        if self._parcels is None:
            self._parcels = DataRecord(
                self._grid,
                items=items,
                time=[time],
                data_vars=variables,
                dummy_elements={"link": [_OUT_OF_NETWORK]},
            )
        else:
            self._parcels.add_item(time=[time], new_item=items, new_item_spec=variables)
            
        return self._parcels

        return self._parcels


def _pulse_characteristics(
    time,
    links,
    n_parcels_at_link,
    total_parcel_volume_at_link,
    max_parcel_volume,
    d50,
    std_dev,
    rho_sediment,
    abrasion_rate,
):

    if n_parcels_at_link is None:  # n_parcels = f(existing parcel volume)
        n_parcels_at_link = np.ceil(
            total_parcel_volume_at_link / max_parcel_volume
        ).astype(dtype=int)
    else:
        n_parcels_at_link, _ = np.broadcast_arrays(n_parcels_at_link, links)

    element_id = np.empty(np.sum(n_parcels_at_link), dtype=int)

    volume = np.full_like(element_id, max_parcel_volume, dtype=float)
    grain_size = np.empty_like(element_id, dtype=float)
    offset = 0
    for link, n_parcels in enumerate(n_parcels_at_link):
        element_id[offset:offset + n_parcels] = links[link]
        grain_size[offset:offset + n_parcels] = np.random.lognormal(
            np.log(d50[link]), np.log(std_dev), n_parcels
        )
        volume[offset] = total_parcel_volume_at_link[link] % n_parcels
        offset += n_parcels
    starting_link = element_id.copy()
    abrasion_rate = np.full_like(element_id, abrasion_rate, dtype=float)
    density = np.full_like(element_id, rho_sediment, dtype=float)

    element_id = np.expand_dims(element_id, axis=1)
    grain_size = np.expand_dims(grain_size, axis=1)
    volume = np.expand_dims(volume, axis=1)

    time_arrival_in_link = np.full(np.shape(element_id), time, dtype=float)
    location_in_link = np.expand_dims(np.random.rand(np.sum(n_parcels_at_link)), axis=1)
    # volume = np.full(np.shape(element_id), 0.05)  # (m3) the volume of each parcel

    # a lithology descriptor for each parcel
    # lithology = ["pulse_material"] * np.size(element_id)

    # 1 = active/surface layer; 0 = subsurface layer
    active_layer = np.ones(np.shape(element_id))

    grid_element = ["link"]*np.size(element_id)
    grid_element = np.expand_dims(grid_element, axis=1)
    

    return {
        "starting_link": (["item_id"], starting_link),
        "abrasion_rate": (["item_id"], abrasion_rate),
        "density": (["item_id"], density),
        # "lithology": (["item_id"], lithology),
        "time_arrival_in_link": (["item_id", "time"], time_arrival_in_link),
        "active_layer": (["item_id", "time"], active_layer),
        "location_in_link": (["item_id", "time"], location_in_link),
        "D": (["item_id", "time"], grain_size),
        "volume": (["item_id", "time"], volume),
    }, {"grid_element": grid_element, "element_id": element_id}



def _pulse_characteristics_user_defined(parcelDF,parcel_vol,time,
                                        point_pulse = True):
    '''
    
    specify attributes of parcels added to the datarecord of an instance of 
    the network sediment transport component. 
    

    Parameters
    ----------
    parcelDF : pandas dataframe
        each row contains information for a single parcel. Includes the 
        following columns:'mw_unit', 'vol [m^3]','raster_grid_cell_#', 'link_#',
        'link_cell_#','raster_grid_to_link_offset [m]','link_downstream_distance'
    
    parcel_vol : float
        volume of each parcel, a single value, used to divide the volume of the pulse into parcels
    time : integer or datetime64 value equal to nst.time
        time that the pulse is triggered in the network sediment transporter

    Returns
    -------
    tuple: (item_id,variables)
        item_id: dictionary, model grid element and index of element of each parcel
        variables: dictionary, variable values for al new pulses
 


    '''
    #(1) create parcels for each landslide pulse
    p_np = []
    for index, row in mwlinkDF.iterrows():
        p_np.append(int(row['vol [m^3]']/p_parcel_vol)) #number of parcels in pulse = volume pulse/volume 1 parcel
 
    num_pulse_parcels = sum(p_np) # total number of parcels that enter network for timestep t

   
    LinkDistanceRatio = np.array([]) #create 1 x num_pulse_parcels array that lists distance ratio of each link. 
    for i,val in enumerate(mwlinkDF['link_downstream_distance'].values):
        # print(val)
        if point_pulse:
            LinkDistanceRatio = np.concatenate((LinkDistanceRatio,np.ones(p_np[i])*val)) #enter channel at single point
        else:
            LinkDistanceRatio = np.concatenate((LinkDistanceRatio,np.linspace(val,0.99,p_np[i]))) #enter channel distributed from deposition point to end of link
    
    new_location_in_link = np.expand_dims(LinkDistanceRatio, axis=1)
    
    #(3)create 1xnum_pulse_parcels array that lists the link each parcel is entered into.
    newpar_element_id = np.array([]) 
    for i, row in mwlinkDF.iterrows():       
        newpar_element_id = np.concatenate((newpar_element_id,np.ones(p_np[i])*row['link_#']))        

    newpar_element_id = np.expand_dims(newpar_element_id.astype(int), axis=1) #change format to 1Xn array
    new_starting_link = np.squeeze(newpar_element_id)

    #(4)create 1xn array of zeros to append to array of distance parcels traveled before tiemestep t (zero because parcels did not exist before timestep)
    newpar_dist = np.zeros(num_pulse_parcels,dtype=int)          
         
    
    #(5) create time stamp of zero for each parcel before parcel existed        
    new_time_arrival_in_link = nst._time* np.ones(
        np.shape(newpar_element_id)) #arrives at current time in nst model
    
    #(6) compute total volume of all parcels entered into network during timestep
    new_volume = p_parcel_vol*np.ones(np.shape(newpar_element_id)) #mwlinkDF['vol [m^3]'].values /100  # volume of each parcel (m3) divide by 100 because large parcels break model
    #new_volume = np.expand_dims(new_volume, axis=1)
    
    #(7) assign grain properties -lithology ,activity, density, abrasion rate, diameter,- this can come from dataframe mwlinkDF
    new_lithology = ["pulse_material"] * np.size(
        newpar_element_id)  
    
    new_active_layer = np.ones(
        np.shape(newpar_element_id))  # 1 = active/surface layer; 0 = subsurface layer
    
    new_density = 2650 * np.ones(np.size(newpar_element_id))  # (kg/m3)
        
    new_abrasion_rate = 0 * np.ones(np.size(newpar_element_id))
    
    new_D = p_parcel_D * np.ones(np.shape(newpar_element_id))


    #(8) assign part of grid that parcel is deposited (node vs link)    
    newpar_grid_elements = np.array(
        np.empty(
            (np.shape(newpar_element_id)), dtype=object
        )
    ) 
    
    newpar_grid_elements.fill("link")
    
    item_id = {"grid_element": newpar_grid_elements,
             "element_id": newpar_element_id}

    #(9) construct dictionary of all parcel variables to be entered into data recorder
    variables = {
        "starting_link": (["item_id"], new_starting_link),
        "abrasion_rate": (["item_id"], new_abrasion_rate),
        "density": (["item_id"], new_density),
        #"lithology": (["item_id"], new_lithology),
        "time_arrival_in_link": (["item_id", "time"], new_time_arrival_in_link),
        "active_layer": (["item_id", "time"], new_active_layer),
        "location_in_link": (["item_id", "time"], new_location_in_link),
        "D": (["item_id", "time"], new_D),
        "volume": (["item_id", "time"], new_volume),
    }
    
    return variables,item_id




def _calc_approx_parcel_volume(total_parcel_volume_at_link):
    min_link_volume = np.min(total_parcel_volume_at_link)
    min_number_of_starting_parcels = 100
    return min_link_volume / min_number_of_starting_parcels


def calc_total_parcel_volume(width, length, sediment_thickness):
    return width * length * sediment_thickness


def make_sediment(
    grid,
    time,
    time_arrival_in_link,
    abrasion_rate,
    density,
    active_layer,
    location_in_link,
    D,
    volume,
    number_of_parcels,
):
    """One line description.

    More info goes here.

    Note that this is unit agnostic, but that it is designed to work with
    the :py:class:`~landlab.components.NetworkSedimentTransporter` which
    requires mks units.


    When input is a dictionary it has the form:

    .. code-block:: python

      {"distribution": "name of distribution",
       #other keyword argument for that distribution in numpy random.
       }

    Right now we are only making the core required inputs for
    the NST. But eventually want to be able to expand to other
    arbitratry inputs (e.g., lithology, osl characteristics,
    etc)


    Parameters
    ----------
    grid : model grid
    time : float
        Time at which sediment is created.
    number_of_parcels

    which_links :
        information about which links parcels are placed on (default is all?)

    abrasion_rate : float or dict (see above)
        must be greater than zero
    density : float
        must be greater than zero.

    location_in_link
    D
    volume

    # talk with AMP and JC about what the right way to specify the order of
    # being added.

    # check with AMP regarding which of these are actually required.
    # is time arrival in link just the current model time?

    # need to think about time arrival at link carefully because time arrival
    # at link is what controls when a parcel is brought into the active layer
    # or not. When active, you move downstream,
    # So you can be deep in the link, and close to the end of the link, and not
    # move for a long time.

    # Have eric help with creating additional attributes, he will have good
    # ideas based on his work with layers.

    Examples
    --------
    >>> # from landlab.utils.parcels import make_sediment

    Make one example that uses all default values.

    Make one example that uses all scalars

    Make one example that uses numpy.random

    In unit tests ensure volume is always correct, even with
    wierd distributions.
    Check that thing that are floats are always floats
    Check (same with ints)


    """
    # Part 1:
    # For each required attribute, do something like this to create the
    # required values.

    if isinstance(abrasion_rate, dict):
        distribution = abrasion_rate.pop("distribution")
        function = np.random.__dict__[distribution]
        values = function(size=number_of_parcels, **abrasion_rate)
    else:
        values = abrasion_rate * np.ones(number_of_parcels)

    # Part 2: Some attributes values are pre-constrained.
    # time_arrival_in_link is the time (though need to be careful with FILO)

    # active_layer can be set to _INACTIVE (it will be overwritten in run one step).

    # Part 3: Deal with any additional attributes that are not required:

    # Part 4:
    # given attributes, create the dictionary that data record wants for
    # add items.
    # The following code is pasted directly from Allison's example script.
    # so all the variables/names will need to be created.

    # create new parcel grid elements and element IDs.

    # elements is an array of "link"

    # based on which_links determine which link IDs parcels are placed on.

    # Note that the syntax of how these three variables (new_parcels,
    # new_variables) is very touchy b/c the DataRecord is touchy.
    new_parcels = {
        "grid_element": newpar_grid_elements,
        "element_id": newpar_element_id,
    }

    new_variables = {
        "starting_link": (["item_id"], new_starting_link),
        "abrasion_rate": (["item_id"], new_abrasion_rate),
        "density": (["item_id"], new_density),
        "lithology": (["item_id"], new_lithology),
        "time_arrival_in_link": (["item_id", "time"], new_time_arrival_in_link),
        "active_layer": (["item_id", "time"], new_active_layer),
        "location_in_link": (["item_id", "time"], new_location_in_link),
        "D": (["item_id", "time"], new_grain_size),
        "volume": (["item_id", "time"], new_volume),
    }

    items = {"time": [time], "new_item": new_parcels, "new_item_spec": new_variables}
    return items
