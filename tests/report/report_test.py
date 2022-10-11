from dessia_common.report import Report

report = Report(name_report='LogTest',
                width_line=100,
                core='',
                last_offset=0,
                name='name_not_util')

#from automobile-propre.com
marques = ['Renault', 'Renault', 'Renault',
           'Peugeot', 'Peugeot',
           'Dacia',
           'DS',
           'Polestar']
modeles = ['Zoe', 'Twingo E-Tech', 'Mégane électrique',
           'e-208', 'e-2008',
           'Spring',
           '3 Crossback e-Tense',
           '2']
prix = ['32500€', '21350€', '35200€',
        '33000€', '37100€',
        '19290€',
        '39900€',
        '39900€']
autonomies_min = [171, 190, 300,
                  400, 320,
                  230,
                  341,
                  440]
autonomies_max = [390, 190, 470,
                  400, 320,
                  230,
                  341,
                  540]

titles = ['Marque', 'Modèle', 'Prix', 'Autonomie min', 'Autonomie max']

vehicles = []
for k in range(len(marques)):
    vehicles.append([marques[k], modeles[k], prix[k], autonomies_min[k], autonomies_max[k]])


report.add_title('Electric vehicle')
report.add_subtitle('Mostly french')
report.add_subsubtitle('With another brand')
report.add_text('Here is a table with some electric vehicle, mostly french')
report.add_table(titles, vehicles)
