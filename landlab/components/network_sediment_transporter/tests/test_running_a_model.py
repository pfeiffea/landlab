

from landlab import NetworkModelGrid

from landlab.components import (CzubaNetworkSedimentTransporter,
                                FlowAccumulator,
                                SedimentParcels,
                                )



def test_run_model_works():
    """Test that a model can run as expected."""
    y_of_node = []
    x_of_node = []
    nodes_at_link = [(), ()]

    nmg = NetworkModelGrid(y_of_node, x_of_node, nodes_at_link)
    area = nmg.add_field('area_at...')

    fa = FlowAccumulator(nmg)
    parcels = SedimentParcels(grid, initialization_info_including_future_forcing)

    # parcels have time added as an attribute, when updated, they are
    # added.

    hg = HydraulicGeometry(grid, )
    # width and depth scale at stream gauge.
    # apply to the rest of the newtork.

    nsp = CzubaNetworkSedimentTransporter(grid,
                                          transport_formulation='WilcockCrow',
                                          parcels=parcels,
                                        )
    dt = 100
    for i in range(10):
        # something that updates parcels could happen if necessary.
        #

        # update discharge
        fa.run_one_step()

        # update channel geometry
        hg.run_one_step()

        # move sediment
        nsp.run_one_step(dt)
