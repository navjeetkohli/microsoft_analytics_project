from imports import *
from clean import *

#Plotting and Displaying the Graphs
def graphs():
    cars_data = clean_data()
    q0 = None
    q1 = 'Features vs their Counts '
    q2 = 'Top 10 most present company names in the dataset'
    q3 = 'Top 15 least present company names in the dataset'
    q4 = 'Top 10 years with highest sales'
    q5 = 'Top 10 car company with highest selling price'
    q6 = 'Top 10 car company with highest mileage'
    q7 = 'Top 10 car company with highest engine cc'
    q8 = 'Top 10 car company with highest max power'
    q9 = 'Kilometer driven vs Selling price Scatter Plot'
    q10 = 'Mileage vs Selling price Scatter Plot'
    q11 = 'Engine vs Selling price Scatter Plot'
    q12 = 'Max Power vs Selling price Scatter Plot'
    q13 = 'Heat Map of Dataset'

    op = st.selectbox("Select Option for Visualization", [q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13], index=0)
    
    if op == q1:
        fig = plt.figure(figsize=(7,12))
        gs = fig.add_gridspec(5,3)
        gs.update(hspace=0.5, wspace=0.5)

        pal = sns.dark_palette('#E4EF60', reverse=True)
        pal.insert(0, '#9c9a9a')
        # Row 1
        # Countplot
        ax0 = fig.add_subplot(gs[0:2,0:3])
        ax0_s = sns.countplot(data=cars_data,
                            x='seats', 
                            palette=pal, 
                            edgecolor='k', 
                            linewidth=0.6, 
                            order=[i for i in cars_data.seats.value_counts().index])
        ax0_s.tick_params(labelsize=5, left=False)
        ax0_s.set_xlabel(xlabel='Seats', fontsize=5, weight='bold')
        ax0_s.set_ylabel(None)

        for p in ax0_s.patches:
            value = f"{p.get_height():,.0f}"
            x = p.get_x() + p.get_width()/2
            y = p.get_y() + p.get_height()+500
            ax0.text(x, y, value, fontsize=4.5, ha='center', va='center', 
                    bbox=dict(facecolor='white', edgecolor='k', boxstyle='round', linewidth=0.5))
            
        # Row 2 
        # Countplot
        ax1 = fig.add_subplot(gs[2,0:2])
        ax1_s = sns.countplot(data=cars_data, y='fuel_type', 
                            palette=pal, edgecolor='k', linewidth=0.6, 
                            order=[i for i in cars_data.fuel_type.value_counts().index])
        ax1_s.tick_params(labelsize=5, left=False)
        ax1_s.set_xlabel(None)
        ax1_s.set_ylabel(ylabel='fuel_type', fontsize=5, weight='bold')
        for p in ax1_s.patches:
            value = f"{p.get_width():,.0f}"
            x = p.get_x() + p.get_width()+500
            y = p.get_y() + p.get_height()/2
            ax1.text(x, y, value, fontsize=5, ha='center', va='center', 
                    bbox=dict(facecolor='white', edgecolor='k', boxstyle='round', linewidth=0.4))
        # Pie plot
        ax11 = fig.add_subplot(gs[2,2])
        ax11.pie(x = cars_data.fuel_type.value_counts().values,
                colors=['#9c9a9a','skyblue','#0fa2bd','#075c91','#012338'],
                explode = [0.03 for i in range(cars_data.fuel_type.nunique())],
                radius=1.2,
                textprops=dict(fontsize=5))
        ax11.add_artist(plt.Circle((0,0),0.4,fc='white'))
        ax11.legend(labels = cars_data.fuel_type.value_counts().index, 
                    ncol=5, fontsize=4,edgecolor='white', shadow=True, loc='lower center',bbox_to_anchor=(0.5,-0.1))

        # Row 3 
        # Countplot
        ax2 = fig.add_subplot(gs[3,0:2])
        ax2_s = sns.countplot(data=cars_data, y='seller_type', 
                            palette=pal, edgecolor='k', linewidth=0.6, 
                            order=[i for i in cars_data.seller_type.value_counts().index])
        ax2_s.tick_params(labelsize=5, left=False)
        ax2_s.set_xlabel(None)
        ax2_s.set_ylabel(ylabel='seller_type', fontsize=5, weight='bold')
        for p in ax2_s.patches:
            value = f"{p.get_width():,.0f}"
            x = p.get_x() + p.get_width()+650
            y = p.get_y() + p.get_height()/2
            ax2.text(x, y, value, fontsize=5, ha='center', va='center', 
                    bbox=dict(facecolor='white', edgecolor='k', boxstyle='round', linewidth=0.4))
        # Pie plot  
        ax21 = fig.add_subplot(gs[3,2])
        ax21.pie(x = cars_data.seller_type.value_counts().values,
                colors=['#9c9a9a','skyblue','#0fa2bd','#075c91','#012338'],
                explode = [0.03 for i in range(cars_data.seller_type.nunique())],
                textprops=dict(fontsize=5),
                radius=1.2)
        ax21.add_artist(plt.Circle((0,0),0.4,fc='white'))
        ax21.legend(labels = cars_data.seller_type.value_counts().index, 
                    ncol=3, fontsize=4,edgecolor='white', shadow=True, loc='lower center',bbox_to_anchor=(0.5,-0.1))

        # Row 4 
        # Countplot
        ax3 = fig.add_subplot(gs[4,0:2])
        ax3_s = sns.countplot(data=cars_data, y='transmission_type', 
                            palette=pal, edgecolor='k', linewidth=0.6, 
                            order=[i for i in cars_data.transmission_type.value_counts().index])
        ax3_s.tick_params(labelsize=5, left=False)
        ax3_s.set_xlabel(None)
        ax3_s.set_ylabel(ylabel='transmission_type', fontsize=5, weight='bold')
        for p in ax3_s.patches:
            value = f"{p.get_width():,.0f}"
            x = p.get_x() + p.get_width()+850
            y = p.get_y() + p.get_height()/2
            ax3.text(x, y, value, fontsize=5, ha='center', va='center', 
                    bbox=dict(facecolor='white', edgecolor='k', boxstyle='round', linewidth=0.4))
        # Pie plot
        ax31 = fig.add_subplot(gs[4,2])
        ax31.pie(x = cars_data.transmission_type.value_counts().values,
                colors=['#9c9a9a','skyblue','#0fa2bd','#075c91','#012338'],
                explode = [0.03 for i in range(cars_data.transmission_type.nunique())],
                textprops=dict(fontsize=5),
                radius=1.2)
        ax31.add_artist(plt.Circle((0,0),0.4,fc='white'))
        ax31.legend(labels = cars_data.transmission_type.value_counts().index, 
                    ncol=2, edgecolor='white',shadow=True, fontsize=4, loc='lower center',bbox_to_anchor=(0.5,-0.1))

        sns.despine()
        st.pyplot(fig)

    if op == q2:
        fig = plt.figure(figsize=(7,3))
        gs = fig.add_gridspec(1,1)

        cars = cars_data.company.value_counts().head(10)
        pal = sns.light_palette('#E4EF60', n_colors=50, reverse=True)
        pal[0] = '#9c9a9a'
        ax = fig.add_subplot(gs[0,0])
        ax=sns.barplot(x=cars.index, y=cars.values, edgecolor='k', linewidth=0.7, palette=pal)
        ax.tick_params(axis='x', labelsize=5, rotation=90)
        ax.tick_params(axis='y', labelsize=5, left=False)
        ax.set_ylabel(ylabel='count', fontsize=5, weight='bold')

        for p in ax.patches:
            value = f'{p.get_height():,.0f}'
            x = p.get_x() + p.get_width()/2
            y = p.get_y() + p.get_height()+320
            ax.text(x=x, y=y, s=value, fontsize=4.5, ha='center', va='center',
                bbox=dict(fc='white', edgecolor='k', boxstyle='round', linewidth=0.5))
            
        fig.suptitle(t='Top 10 most present company names in the dataset', fontsize='6', weight='bold')
        sns.despine()
        st.pyplot(fig)

    if op == q3:
        fig = plt.figure(figsize=(7,3))
        gs = fig.add_gridspec(1,1)

        cars = cars_data.company.value_counts().tail(15)
        pal = sns.light_palette('#E4EF60', n_colors=50, reverse=True)
        pal[0] = '#9c9a9a'
        ax = fig.add_subplot(gs[0,0])
        ax=sns.barplot(x=cars.index, y=cars.values, edgecolor='k', linewidth=0.7, palette=pal)
        ax.tick_params(axis='x', labelsize=5, rotation=90)
        ax.tick_params(axis='y', labelsize=5, left=False)
        ax.set_ylabel(ylabel='count', fontsize=5, weight='bold')

        for p in ax.patches:
            value = f'{p.get_height():,.0f}'
            x = p.get_x() + p.get_width()/2
            y = p.get_y() + p.get_height()+0.5
            ax.text(x=x, y=y, s=value, fontsize=4.5, ha='center', va='center',
                bbox=dict(fc='white', edgecolor='k', boxstyle='round', linewidth=0.5))
            
        fig.suptitle(t='Top 15 least present company names in the dataset', fontsize='6', weight='bold')
        sns.despine()
        st.pyplot(fig)

    if op == q4:
        fig = plt.figure(figsize=(7,3))
        gs = fig.add_gridspec(1,1)

        cars = cars_data.year.value_counts().head(10)
        pal = sns.light_palette('#E4EF60', n_colors=50, reverse=True)
        pal[7] = '#9c9a9a'
        ax = fig.add_subplot(gs[0,0])
        ax=sns.barplot(x=cars.index, y=cars.values, edgecolor='k', linewidth=0.7, palette=pal)
        ax.tick_params(axis='x', labelsize=5, rotation=90)
        ax.tick_params(axis='y', labelsize=5, left=False)
        ax.set_ylabel(ylabel='count', fontsize=5, weight='bold')

        for p in ax.patches:
            value = f'{p.get_height():,.0f}'
            x = p.get_x() + p.get_width()/2
            y = p.get_y() + p.get_height()+150
            ax.text(x=x, y=y, s=value, fontsize=4.5, ha='center', va='center',
                bbox=dict(fc='white', edgecolor='k', boxstyle='round', linewidth=0.5))
            
        fig.suptitle(t='Top 10 years with highest sales', fontsize='6', weight='bold')
        sns.despine()
        st.pyplot(fig)

    if op == q5:
        fig = plt.figure(figsize=(7,3))
        gs = fig.add_gridspec(1,1)

        pal = sns.light_palette('#E4EF60', reverse=True, n_colors=50)
        pal.insert(0,'#9c9a9a')

        cars = cars_data.groupby('company')['selling_price'].mean().sort_values(ascending=False).head(10)/1000000
        ax = fig.add_subplot(gs[0,0])
        ax=sns.barplot(x=cars.index, y=cars.values, edgecolor='k', linewidth=0.7, palette=pal)
        ax.tick_params(axis='x', labelsize=5, rotation=90)
        ax.tick_params(axis='y', labelsize=5, left=False)
        ax.set_xlabel(xlabel=None)
        ax.set_ylabel(ylabel='price in millions', fontsize=5, weight='bold')
        for p in ax.patches:
            value = f'{p.get_height():,.1f}' + ' M'
            x = p.get_x() + p.get_width()/2
            y = p.get_y() + p.get_height()+3
            ax.text(x=x, y=y, s=value, fontsize=4.5, ha='center', va='center',
                bbox=dict(fc='white', edgecolor='k', boxstyle='round', linewidth=0.5))
            
        fig.suptitle(t='Top 10 car company with highest selling price', fontsize=6, weight='bold')
        sns.despine()
        st.pyplot(fig)

    if op == q6:
        fig = plt.figure(figsize=(7,3))
        gs = fig.add_gridspec(1,1)

        pal = sns.light_palette('#E4EF60', reverse=True, n_colors=50)
        pal.insert(0,'#9c9a9a')

        cars = cars_data.groupby('company')['mileage'].mean().sort_values(ascending=False).head(10)
        ax = fig.add_subplot(gs[0,0])
        ax=sns.barplot(x=cars.index, y=cars.values, edgecolor='k', linewidth=0.7, palette=pal)
        ax.tick_params(axis='x', labelsize=5, rotation=90)
        ax.tick_params(axis='y', labelsize=5, left=False)
        ax.set_xlabel(xlabel=None)
        ax.set_ylabel(ylabel='mileage in kmpl', fontsize=5, weight='bold')
        ax.set_ylim(17,22.5)

        for p in ax.patches:
            value = f'{p.get_height():,.1f}'
            x = p.get_x() + p.get_width()/2
            y = p.get_y() + p.get_height()+0.3
            ax.text(x=x, y=y, s=value, fontsize=4.5, ha='center', va='center',
                bbox=dict(fc='white', edgecolor='k', boxstyle='round', linewidth=0.5))
            
        fig.suptitle(t='Top 10 car company with highest mileage', fontsize=6, weight='bold')
        sns.despine()
        st.pyplot(fig)

    if op == q7:
        fig = plt.figure(figsize=(7,3))
        gs = fig.add_gridspec(1,1)

        pal = sns.light_palette('#E4EF60', reverse=True, n_colors=50)
        pal.insert(0,'#9c9a9a')

        cars = cars_data.groupby('company')['engine'].mean().sort_values(ascending=False).head(10)
        ax = fig.add_subplot(gs[0,0])
        ax=sns.barplot(x=cars.index, y=cars.values, edgecolor='k', linewidth=0.7, palette=pal)
        ax.tick_params(axis='x', labelsize=5, rotation=90)
        ax.tick_params(axis='y', labelsize=5, left=False)
        ax.set_xlabel(xlabel=None)
        ax.set_ylabel(ylabel='engine in cc', fontsize=5, weight='bold')
        for p in ax.patches:
            value = f'{p.get_height():,.0f}'
            x = p.get_x() + p.get_width()/2
            y = p.get_y() + p.get_height()+400
            ax.text(x=x, y=y, s=value, fontsize=4.5, ha='center', va='center',
                bbox=dict(fc='white', edgecolor='k', boxstyle='round', linewidth=0.5))
            
        fig.suptitle(t='Top 10 car company with highest engine cc', fontsize=6, weight='bold')
        sns.despine()
        st.pyplot(fig)

    if op == q8:
        fig = plt.figure(figsize=(7,3))
        gs = fig.add_gridspec(1,1)

        pal = sns.light_palette('#E4EF60', reverse=True, n_colors=50)
        pal.insert(0,'#9c9a9a')

        cars = cars_data.groupby('company')['max_power'].mean().sort_values(ascending=False).head(10)
        ax = fig.add_subplot(gs[0,0])
        ax=sns.barplot(x=cars.index, y=cars.values, edgecolor='k', linewidth=0.7, palette=pal)
        ax.tick_params(axis='x', labelsize=5, rotation=90)
        ax.tick_params(axis='y', labelsize=5, left=False)
        ax.set_xlabel(xlabel=None)
        ax.set_ylabel(ylabel='max power in bhp', fontsize=5, weight='bold')
        for p in ax.patches:
            value = f'{p.get_height():,.0f}'
            x = p.get_x() + p.get_width()/2
            y = p.get_y() + p.get_height()+38
            ax.text(x=x, y=y, s=value, fontsize=4.5, ha='center', va='center',
                bbox=dict(fc='white', edgecolor='k', boxstyle='round', linewidth=0.5))
            
        fig.suptitle(t='Top 10 car company with highest max power', fontsize=6, weight='bold')
        sns.despine()
        st.pyplot(fig)

    if op == q9:
        fig = plt.figure(figsize=(7,5))
        gs = fig.add_gridspec(2,1)
        color = ['#79c6e8' for i in range(cars_data.shape[0])]
        color[4967] = color[19972] = color[14257] = color[475] = '#9c9a9a'
        ax=fig.add_subplot(gs[0,0])
        ax.scatter(x=cars_data.km_driven/1000000,y=cars_data.selling_price/1000000,s=28,
                            color=color, edgecolor='white', linewidth=0.4)
        ax.tick_params(labelsize=5, left=False)
        ax.set_ylabel(ylabel='selling price in millions', fontsize=5, weight='bold')

        ax1=fig.add_subplot(gs[1,0])
        rmd_data = cars_data[cars_data['selling_price']<20000000]
        rmd_data = rmd_data[rmd_data['km_driven']<2000000]
        ax1.scatter(x=rmd_data.km_driven/1000000,y=rmd_data.selling_price/1000000,
                    s=28, color='#79c6e8', edgecolor='white', linewidth=0.4)
        ax1.tick_params(labelsize=5, left=False)
        ax1.set_xlabel(xlabel='kilometer driven in millions', fontsize=5, weight='bold')
        ax1.set_ylabel(ylabel='selling price in millions', fontsize=5, weight='bold')
        ax1.text(x=1, y=12, s='After removing outliers', fontsize=6, weight='bold', alpha=0.6,
                bbox=dict(facecolor='white', lw=0.5, alpha=0.3))
        sns.despine()
        st.pyplot(fig)

    if op == q10:
        fig = plt.figure(figsize=(7,5))
        gs = fig.add_gridspec(2,1)
        color = ['#79c6e8' for i in range(cars_data.shape[0])]
        color[4967] = color[14257] = color[475] = color[11964] = color[18396]= '#9c9a9a'
        ax=fig.add_subplot(gs[0,0])
        ax.scatter(x=cars_data.mileage,y=cars_data.selling_price/1000000,s=28,
                            color=color, edgecolor='white', linewidth=0.4)
        ax.tick_params(labelsize=5, left=False)
        ax.set_ylabel(ylabel='selling price in millions', fontsize=5, weight='bold')

        ax1=fig.add_subplot(gs[1,0])
        rmd_data = cars_data[cars_data['selling_price']<20000000]
        rmd_data = rmd_data[rmd_data['mileage']<100]
        ax1.scatter(x=rmd_data.mileage,y=rmd_data.selling_price/1000000,s=28,
                            color='#79c6e8', edgecolor='white', linewidth=0.4)
        ax1.tick_params(labelsize=5, left=False)
        ax1.set_xlabel(xlabel='mileage in kmpl', fontsize=5, weight='bold')
        ax1.set_ylabel(ylabel='selling price in millions', fontsize=5, weight='bold')
        ax1.text(x=25, y=12, s='After removing outliers', fontsize=6, weight='bold', alpha=0.6,
                bbox=dict(facecolor='white', lw=0.5, alpha=0.3))
        sns.despine()
        st.pyplot(fig)

    if op == q11:
        fig = plt.figure(figsize=(7,5))
        gs = fig.add_gridspec(2,1)
        color = ['#79c6e8' for i in range(cars_data.shape[0])]
        color[4967] = color[475] = color[14257] = '#9c9a9a'
        ax=fig.add_subplot(gs[0,0])
        ax.scatter(x=cars_data.engine,y=cars_data.selling_price/1000000,s=28,
                            color=color, edgecolor='white', linewidth=0.4)
        ax.tick_params(labelsize=5, left=False)
        ax.set_ylabel(ylabel='selling price in millions', fontsize=5, weight='bold')

        ax1=fig.add_subplot(gs[1,0])
        rmd_data = cars_data[cars_data['selling_price']<20000000]
        rmd_data = rmd_data[rmd_data['engine']<6100]
        ax1.scatter(x=rmd_data.engine,y=rmd_data.selling_price/1000000,s=28,
                            color='#79c6e8', edgecolor='white', linewidth=0.4)
        ax1.tick_params(labelsize=5, left=False)
        ax1.set_xlabel(xlabel='engine in cc', fontsize=5, weight='bold')
        ax1.set_ylabel(ylabel='selling price in millions', fontsize=5, weight='bold')
        ax1.text(x=4500, y=11, s='After removing outliers', fontsize=6, weight='bold', alpha=0.6,
                bbox=dict(facecolor='white', lw=0.5, alpha=0.3))
        sns.despine()
        st.pyplot(fig)

    if op == q12:
        fig = plt.figure(figsize=(7,5))
        gs = fig.add_gridspec(2,1)
        color = ['#79c6e8' for i in range(cars_data.shape[0])]
        color[4967]=color[475]=color[14257]=color[1536]=color[17020]=color[16856]=color[13130]=color[3980]='#9c9a9a'
        ax=fig.add_subplot(gs[0,0])
        ax.scatter(x=cars_data.max_power,y=cars_data.selling_price/1000000,s=28,
                            color=color, edgecolor='white', linewidth=0.4)
        ax.tick_params(labelsize=5, left=False)
        ax.set_ylabel(ylabel='selling price in millions', fontsize=5, weight='bold')

        ax1=fig.add_subplot(gs[1,0])
        rmd_data = cars_data[cars_data['selling_price']<20000000]
        rmd_data = rmd_data[rmd_data['max_power']<530]
        ax1.scatter(x=rmd_data.max_power,y=rmd_data.selling_price/1000000,s=28,
                            color='#79c6e8', edgecolor='white', linewidth=0.4)
        ax1.tick_params(labelsize=5, left=False)
        ax1.set_xlabel(xlabel='max_power in bhp', fontsize=5, weight='bold')
        ax1.set_ylabel(ylabel='selling price in millions', fontsize=5, weight='bold')
        ax1.text(x=380, y=11, s='After removing outliers', fontsize=6, weight='bold', alpha=0.6,
                bbox=dict(facecolor='white', lw=0.5, alpha=0.3))
        sns.despine()
        st.pyplot(fig)

    if op == q13:
        fig = plt.figure(figsize=(6,4))
        ax = sns.heatmap(cars_data.corr(), cmap='Blues', mask=np.triu(cars_data.corr(), k=1), cbar=False, annot=True,
                        annot_kws=dict(fontsize=4))
        ax.set_facecolor('white')
        ax.tick_params(labelsize=5)
        st.pyplot(fig)

