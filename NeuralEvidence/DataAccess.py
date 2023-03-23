'''
This is a script to extract spike data from Allen Brain Observatory dataset
Environment:
source activate allensdk
'''
import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

data_directory = '/nadata/cnl/data/yuchen/HCDecode/Allen'
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)


# get sessions data (note the filter)
sessions = cache.get_session_table()
print('Total number of sessions: ' + str(len(sessions)))
CA_VISp_sessions = sessions[(sessions.full_genotype.str.find('wt/wt') > -1) & \
                      (sessions.session_type == 'functional_connectivity') & \
                      (['CA3' in acronyms and 'CA1' in acronyms and 'VISp' in acronyms and 'DG' in acronyms for acronyms in sessions.ecephys_structure_acronyms])]


# probes = cache.get_probes()
# channels = cache.get_channels()
# units = cache.get_units()



# download data and save as txt
region_name = ['CA1','CA3','DG','VISp']
sti_name = ['natural_movie_one_more_repeats','natural_movie_one_shuffled','gabors','flashes','drifting_gratings_contrast','drifting_gratings_75_repeats','dot_motion']

for session_idx in range(len(CA_VISp_sessions)):
  session_id = CA_VISp_sessions.index.values[session_idx]
  session = cache.get_session_data(session_id,
                                   isi_violations_maximum = 0.5,
                                   amplitude_cutoff_maximum = 0.1,
                                   presence_ratio_minimum = 0.9)
  for sti in sti_name:
    movie = session.get_stimulus_table(sti)
    movie.to_csv(data_directory+'/session_'+str(session_id)+'/'+sti+'_info.csv')
    movie['frame'].to_csv(data_directory+'/session_'+str(session_id)+'/'+sti+'.csv',header=False)
    for region in region_name:
      units = session.units[session.units["ecephys_structure_acronym"] == region]
      spikes = session.presentationwise_spike_times(
          stimulus_presentation_ids=movie.index.values,
          unit_ids=units.index.values[:])
      spikes.head().to_csv(data_directory+'/session_'+str(session_id)+'/'+sti+'_spikes_'+region+'_info.csv')
      spikes.to_csv(data_directory+'/session_'+str(session_id)+'/'+sti+'_spikes_'+region+'.csv',header=False)

