import { html, render } from 'lit-html';
import * as lit_components from 'lit-components';
import LITApp from '@tensorflow/lit-app';

//# Define a new tab that displays the filtered test data.
//filtered_data_tab = lit_html.Tab(
//'Filtered Test Data',
//lit_components.DataTable(
//examples=[],
//column_names=['text', 'label', 'bert_score'],
//selectable=False,
//sort_by='bert_score'
//),
//data_fetcher=filter_data_callback
//)
//
//Add the new tab to the LIT app.
//lit_app = LITApp(
//model_type='classification',
//model_path=model_path,
//dataset=dataset,
//# ... other arguments ...
//tabs=[predictions_tab, filtered_data_tab]
//)

function create_app(model_path, dataset, filter_data_callback) {
  // Define the template for the application
  const template = () => html`
    <lit-app
      .app=${{
        modelType: 'classification',
        modelPath: model_path,
        dataset: dataset,
        tabs: [
          {
            title: 'Predictions',
            type: 'predictions',
          },
          {
            title: 'Filtered Test Data',
            type: 'custom',
            component: lit_components.DataTable,
            args: {
              examples: [],
              columnNames: ['text', 'label', 'bert_score'],
              selectable: false,
              sortBy: 'bert_score'
            },
            dataFetcher: filter_data_callback
          }
        ]
      }}
    ></lit-app>
  `;

  // Initialize the application
  const app = document.getElementById('app');
  render(template(), app);
}

// Call the create_app() function with the necessary parameters
create_app('path/to/model', 'my_dataset', filter_data_callback);