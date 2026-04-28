/**
 * App — interactive XOR neural-network playground.
 *
 * State flow:
 *   network          — current weights & activations (drives diagram + tables)
 *   inputX1 / inputX2 — slider values (0–1) for live prediction
 *   training         — boolean lock while SGD is running
 *   lossHistory      — array of MSE values for the canvas chart
 *   xorPredictions   — current network output on the four XOR cases
 *
 * Training uses requestAnimationFrame so the UI updates smoothly in
 * 100-epoch chunks without blocking the main thread.
 */

import { type ReactNode, useEffect, useRef, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import CodeExplorer from "./components/CodeExplorer";
import GridCounter from "./components/GridCounter";
import Matrix3DWorld from "./components/Matrix3DWorld";
import MatrixMultiplyDemo from "./components/MatrixMultiplyDemo";
import NetworkDiagram from "./components/NetworkDiagram";
import WhyMatrix from "./components/WhyMatrix";
import {
	createNetwork,
	type Layer,
	matFrom,
	predict,
	randomizeNetwork,
	train,
	xorInputs,
	xorTargets,
} from "./nn-engine";
import "./App.css";

const INITIAL_EPOCHS = 10000;
const LR = 0.1;

function App() {
	const { t, i18n } = useTranslation();
	const [mode, setMode] = useState<"xor" | "grid">("xor");
	const [network, setNetwork] = useState(() => {
		const net = createNetwork([2, 4, 1]);
		randomizeNetwork(net, 42);
		return net;
	});
	const [gridNetwork, setGridNetwork] = useState(() => {
		const net = createNetwork([100, 64, 1]);
		randomizeNetwork(net, 42);
		return net;
	});
	const [inputX1, setInputX1] = useState(0);
	const [inputX2, setInputX2] = useState(0);
	const [prediction, setPrediction] = useState(0);
	const [training, setTraining] = useState(false);
	const [epochsDone, setEpochsDone] = useState(0);
	const [lossHistory, setLossHistory] = useState<number[]>([]);
	const [xorPredictions, setXorPredictions] = useState<number[]>([]);
	const animRef = useRef<number | null>(null);

	useEffect(() => {
		const input = matFrom(2, 1, [inputX1, inputX2]);
		setPrediction(predict(network, input).data[0]);
		setXorPredictions(xorInputs.map((inp) => predict(network, inp).data[0]));
	}, [network, inputX1, inputX2]);

	const startTraining = () => {
		if (training) return;
		setTraining(true);
		setLossHistory([]);
		setEpochsDone(0);

		const net = createNetwork([2, 4, 1]);
		randomizeNetwork(net, 42);

		let currentEpoch = 0;
		const batchSize = 100;

		const step = () => {
			const losses: number[] = [];
			train(net, xorInputs, xorTargets, batchSize, LR, (_epoch, avgLoss) => {
				losses.push(avgLoss);
			});
			currentEpoch += batchSize;

			setNetwork({ layers: net.layers.map((l) => ({ ...l })) });
			setEpochsDone(currentEpoch);
			setLossHistory((prev) => [...prev, ...losses]);
			setXorPredictions(xorInputs.map((inp) => predict(net, inp).data[0]));

			if (currentEpoch < INITIAL_EPOCHS) {
				animRef.current = requestAnimationFrame(step);
			} else {
				setTraining(false);
				animRef.current = null;
			}
		};

		animRef.current = requestAnimationFrame(step);
	};

	const resetNetwork = () => {
		if (animRef.current) {
			cancelAnimationFrame(animRef.current);
			animRef.current = null;
		}
		const net = createNetwork([2, 4, 1]);
		randomizeNetwork(net, 42);
		setNetwork(net);
		setTraining(false);
		setEpochsDone(0);
		setLossHistory([]);
		setXorPredictions(xorInputs.map((inp) => predict(net, inp).data[0]));
	};

	return (
		<div className="min-h-screen bg-slate-50 text-slate-900">
			<div className="max-w-5xl mx-auto px-6 py-10">
				<header className="mb-10 flex items-start justify-between">
					<div>
						<h1 className="text-3xl font-bold mb-2">{t("app.title")}</h1>
						<p className="text-slate-600">{t("app.subtitle")}</p>
					</div>
					<div className="flex gap-2">
						<button
							type="button"
							onClick={() => setMode("xor")}
							className={`px-3 py-1 rounded text-sm font-medium transition ${
								mode === "xor"
									? "bg-blue-600 text-white"
									: "bg-slate-200 text-slate-700 hover:bg-slate-300"
							}`}
						>
							XOR
						</button>
						<button
							type="button"
							onClick={() => setMode("grid")}
							className={`px-3 py-1 rounded text-sm font-medium transition ${
								mode === "grid"
									? "bg-blue-600 text-white"
									: "bg-slate-200 text-slate-700 hover:bg-slate-300"
							}`}
						>
							Grid
						</button>
						<button
							type="button"
							onClick={() => i18n.changeLanguage("en")}
							className={`px-3 py-1 rounded text-sm font-medium transition ${
								i18n.language === "en"
									? "bg-blue-600 text-white"
									: "bg-slate-200 text-slate-700 hover:bg-slate-300"
							}`}
						>
							EN
						</button>
						<button
							type="button"
							onClick={() => i18n.changeLanguage("vi")}
							className={`px-3 py-1 rounded text-sm font-medium transition ${
								i18n.language === "vi"
									? "bg-blue-600 text-white"
									: "bg-slate-200 text-slate-700 hover:bg-slate-300"
							}`}
						>
							VI
						</button>
					</div>
				</header>

				<div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5 mb-8 text-slate-700 text-sm leading-relaxed">
					<h2 className="text-base font-semibold text-slate-800 mb-2">
						{t("app.whatIsThis")}
					</h2>
					<p className="mb-2">
						<Trans
							i18nKey="app.description1"
							components={{ 1: <strong />, 2: <em /> }}
						/>
					</p>
					<p className="mb-2">
						<Trans
							i18nKey="app.description2"
							components={{
								1: <strong />,
								2: <strong />,
								3: <strong />,
								4: <strong />,
							}}
						/>
					</p>
					<p>{t("app.description3")}</p>
				</div>

				<WhyMatrix />

				<MatrixMultiplyDemo />

				<Matrix3DWorld />

				<div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
					{/* Network diagram */}
					<div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
						<h2 className="text-lg font-semibold mb-4">
							{t("network.architecture")}
						</h2>
						<NetworkDiagram
							layers={network.layers}
							inputValues={[inputX1, inputX2]}
						/>
						<div className="mt-3 flex items-center gap-4 text-sm text-slate-600">
							<span className="flex items-center gap-1">
								<span className="inline-block w-3 h-1 bg-green-500 rounded" />
								{t("network.positiveWeight")}
							</span>
							<span className="flex items-center gap-1">
								<span className="inline-block w-3 h-1 bg-red-500 rounded" />
								{t("network.negativeWeight")}
							</span>
						</div>
					</div>

					{/* Controls */}
					<div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5 space-y-6">
						<h2 className="text-lg font-semibold">
							{t("controls.interactiveInput")}
						</h2>

						<div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-sm text-amber-900">
							<p className="font-semibold mb-1">{t("controls.howItWorks")}</p>
							<p className="mb-2">{t("controls.howItWorksLine1")}</p>
							<ul className="list-disc pl-4 space-y-1">
								<li>
									<Trans
										i18nKey="controls.howItWorksLine2"
										components={{ 1: <strong /> }}
									/>
								</li>
								<li>
									<Trans
										i18nKey="controls.howItWorksLine3"
										components={{ 1: <strong /> }}
									/>
								</li>
								<li>
									<Trans
										i18nKey="controls.howItWorksLine4"
										components={{ 1: <strong /> }}
									/>
								</li>
								<li>
									<Trans
										i18nKey="controls.howItWorksLine5"
										components={{ 1: <strong /> }}
									/>
								</li>
							</ul>
						</div>

						<div>
							<label htmlFor="x1" className="block text-sm font-medium mb-2">
								{t("controls.inputX", { idx: "₁", val: inputX1.toFixed(2) })}
							</label>
							<input
								id="x1"
								type="range"
								min={0}
								max={1}
								step={0.01}
								value={inputX1}
								onChange={(e) => setInputX1(Number(e.target.value))}
								className="w-full accent-blue-600"
							/>
							<div className="flex justify-between text-xs text-slate-400 mt-1">
								<span>0</span>
								<span>1</span>
							</div>
						</div>

						<div>
							<label htmlFor="x2" className="block text-sm font-medium mb-2">
								{t("controls.inputX", { idx: "₂", val: inputX2.toFixed(2) })}
							</label>
							<input
								id="x2"
								type="range"
								min={0}
								max={1}
								step={0.01}
								value={inputX2}
								onChange={(e) => setInputX2(Number(e.target.value))}
								className="w-full accent-blue-600"
							/>
							<div className="flex justify-between text-xs text-slate-400 mt-1">
								<span>0</span>
								<span>1</span>
							</div>
						</div>

						<div className="p-4 bg-slate-100 rounded-lg">
							<div className="text-sm text-slate-600 mb-1">
								{t("controls.prediction")}
							</div>
							<div className="text-2xl font-bold text-blue-700">
								{prediction.toFixed(4)}
							</div>
							<div className="text-xs text-slate-500 mt-1">
								{prediction > 0.5 ? t("controls.class1") : t("controls.class0")}
							</div>
						</div>

						<div className="flex gap-3">
							<button
								type="button"
								onClick={startTraining}
								disabled={training}
								className="flex-1 px-4 py-2.5 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
							>
								{training
									? t("controls.trainingEpochs", { count: epochsDone })
									: t("controls.trainOnXor")}
							</button>
							<button
								type="button"
								onClick={resetNetwork}
								disabled={training}
								className="px-4 py-2.5 bg-slate-200 text-slate-700 rounded-lg font-medium hover:bg-slate-300 disabled:opacity-50 disabled:cursor-not-allowed transition"
							>
								{t("controls.reset")}
							</button>
						</div>
					</div>
				</div>

				{/* XOR Truth Table */}
				<div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5 mb-6">
					<h2 className="text-lg font-semibold mb-4">{t("xor.truthTable")}</h2>
					<div className="grid grid-cols-2 md:grid-cols-4 gap-4">
						{xorInputs.map((inp, idx) => (
							<div
								key={`xor-${inp.data[0]}-${inp.data[1]}`}
								className={`p-4 rounded-lg border-2 transition ${
									Math.abs(xorPredictions[idx] - xorTargets[idx].data[0]) < 0.3
										? "border-green-400 bg-green-50"
										: "border-red-200 bg-red-50"
								}`}
							>
								<div className="text-sm text-slate-600 mb-1">
									{inp.data[0]} XOR {inp.data[1]}
								</div>
								<div className="text-xl font-bold text-slate-800">
									{xorPredictions[idx]?.toFixed(4) ?? "—"}
								</div>
								<div className="text-xs text-slate-500 mt-1">
									{t("xor.target", { value: xorTargets[idx].data[0] })}
								</div>
							</div>
						))}
					</div>
				</div>

				{/* Loss curve */}
				{lossHistory.length > 0 && (
					<div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5 mb-6">
						<h2 className="text-lg font-semibold mb-4">{t("loss.title")}</h2>
						<LossChart data={lossHistory} />
					</div>
				)}

				{/* Weight inspector */}
				<div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
					<h2 className="text-lg font-semibold mb-4">{t("weights.title")}</h2>
					<div className="grid grid-cols-1 md:grid-cols-2 gap-6">
						{network.layers.map((layer, idx) => (
							<WeightMatrix
								key={`wm-${layer.weights.rows}x${layer.weights.cols}`}
								layer={layer}
								idx={idx}
							/>
						))}
					</div>
				</div>

				{mode === "grid" && (
					<GridCounter
						network={gridNetwork}
						onNetworkChange={(net) =>
							setGridNetwork({ layers: net.layers.map((l) => ({ ...l })) })
						}
					/>
				)}

				{/* Code explorer */}
				<div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
					<h2 className="text-lg font-semibold mb-4">{t("code.title")}</h2>
					<CodeExplorer />
				</div>
			</div>
		</div>
	);
}

function WeightMatrix({ layer, idx }: { layer: Layer; idx: number }) {
	const { t } = useTranslation();
	const rows: ReactNode[] = [];
	for (let r = 0; r < layer.weights.rows; r++) {
		const cells: ReactNode[] = [];
		for (let c = 0; c < layer.weights.cols; c++) {
			const v = layer.weights.data[r * layer.weights.cols + c];
			cells.push(
				<td
					key={`cell-${r}-${c}`}
					className={`px-2 py-1 text-center font-mono rounded ${
						v > 0 ? "text-green-700 bg-green-50" : "text-red-700 bg-red-50"
					}`}
				>
					{v.toFixed(3)}
				</td>,
			);
		}
		rows.push(<tr key={`row-${r}`}>{cells}</tr>);
	}

	return (
		<div>
			<h3 className="text-sm font-medium text-slate-600 mb-2">
				{t("weights.layerWeights", {
					idx: idx + 1,
					rows: layer.weights.rows,
					cols: layer.weights.cols,
				})}
			</h3>
			<div className="overflow-x-auto">
				<table className="text-xs w-full">
					<tbody>{rows}</tbody>
				</table>
			</div>
		</div>
	);
}

function LossChart({ data }: { data: number[] }) {
	const { t } = useTranslation();
	const canvasRef = useRef<HTMLCanvasElement>(null);

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas || data.length < 2) return;
		const ctx = canvas.getContext("2d");
		if (!ctx) return;

		const w = canvas.width;
		const h = canvas.height;
		const padding = 40;

		ctx.clearRect(0, 0, w, h);

		const maxLoss = Math.max(...data, 0.25);
		const minLoss = Math.min(...data, 0);
		const range = maxLoss - minLoss || 1;

		// Grid
		ctx.strokeStyle = "#e2e8f0";
		ctx.lineWidth = 1;
		for (let i = 0; i <= 4; i++) {
			const y = padding + (i / 4) * (h - 2 * padding);
			ctx.beginPath();
			ctx.moveTo(padding, y);
			ctx.lineTo(w - padding, y);
			ctx.stroke();
		}

		// Line
		ctx.strokeStyle = "#2563eb";
		ctx.lineWidth = 2;
		ctx.beginPath();
		for (let i = 0; i < data.length; i++) {
			const x = padding + (i / (data.length - 1)) * (w - 2 * padding);
			const y = padding + (1 - (data[i] - minLoss) / range) * (h - 2 * padding);
			if (i === 0) ctx.moveTo(x, y);
			else ctx.lineTo(x, y);
		}
		ctx.stroke();

		// Labels
		ctx.fillStyle = "#64748b";
		ctx.font = "11px sans-serif";
		ctx.textAlign = "right";
		for (let i = 0; i <= 4; i++) {
			const y = padding + (i / 4) * (h - 2 * padding);
			const val = maxLoss - (i / 4) * range;
			ctx.fillText(val.toFixed(3), padding - 6, y + 4);
		}

		ctx.textAlign = "center";
		ctx.fillText(t("loss.epoch"), w / 2, h - 6);
	}, [data, t]);

	return (
		<canvas
			ref={canvasRef}
			width={800}
			height={250}
			className="w-full h-auto border border-slate-200 rounded bg-slate-50"
		/>
	);
}

export default App;
