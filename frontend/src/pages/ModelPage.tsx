const ModelPage = () => (
	<div className="container mx-auto p-6">
	  <h1 className="text-3xl font-bold mb-6">Model Details</h1>
	  <section className="mb-8">
		<h2 className="text-2xl font-semibold mb-4">Machine Learning Pipeline</h2>
		<p className="mb-4">
		  The pipeline includes:
		  - **Data Preprocessing**: Noise removal and normalization.
		  - **Feature Selection**: Recursive Feature Elimination to select significant features.
		  - **Model Training**: Using SVM, RF, kNN, and GBDT.
		</p>
	  </section>
	  <section className="mb-8">
		<h2 className="text-2xl font-semibold mb-4">Training Dataset</h2>
		<p className="mb-4">
		  Trained on the <a href="https://physionet.org/content/mitdb/1.0.0/" target="_blank" className="text-blue-600">MIT-BIH Arrhythmia Database</a>, containing 48 half-hour ECG recordings with annotated beats.
		</p>
	  </section>
	  <section className="mb-8">
		<h2 className="text-2xl font-semibold mb-4">Performance Metrics</h2>
		<p className="mb-4">Example metrics (update with actual values):</p>
		<table className="table-auto w-full border-collapse border">
		  <thead>
			<tr className="bg-gray-200">
			  <th className="border p-2">Metric</th>
			  <th className="border p-2">Value</th>
			</tr>
		  </thead>
		  <tbody>
			<tr><td className="border p-2">Accuracy</td><td className="border p-2">92%</td></tr>
			<tr><td className="border p-2">Precision</td><td className="border p-2">90%</td></tr>
			<tr><td className="border p-2">Recall</td><td className="border p-2">91%</td></tr>
			<tr><td className="border p-2">F1-Score</td><td className="border p-2">90.5%</td></tr>
		  </tbody>
		</table>
	  </section>
	</div>
);

export default ModelPage;