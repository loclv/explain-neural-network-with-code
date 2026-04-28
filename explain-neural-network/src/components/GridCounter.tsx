import { type ReactNode, useState } from "react";
import { useTranslation } from "react-i18next";
import type { NeuralNetwork } from "../nn-engine";
import {
	GRID_SIZE,
	generateGridSamples,
	gridRng,
	makeGridInput,
	matFrom,
	predict,
	train,
} from "../nn-engine";

interface Props {
	network: NeuralNetwork;
	onNetworkChange: (net: NeuralNetwork) => void;
}

export default function GridCounter({ network, onNetworkChange }: Props) {
	const { t } = useTranslation();
	const [grid, setGrid] = useState<number[]>(() =>
		Array.from({ length: GRID_SIZE * GRID_SIZE }, () => 0),
	);
	const [prediction, setPrediction] = useState(0);
	const [training, setTraining] = useState(false);
	const [epochsDone, setEpochsDone] = useState(0);
	const [lossHistory, setLossHistory] = useState<number[]>([]);

	const countPoints = () => grid.filter((c) => c === 1).length;

	const updatePrediction = (g: number[]) => {
		const input = matFrom(GRID_SIZE * GRID_SIZE, 1, g);
		const pred = predict(network, input);
		setPrediction(pred.data[0]);
	};

	const toggleCell = (idx: number) => {
		const next = [...grid];
		next[idx] = next[idx] ? 0 : 1;
		setGrid(next);
		updatePrediction(next);
	};

	const randomize = (points: number) => {
		const rng = gridRng(Date.now());
		const input = makeGridInput(rng, points);
		const next = Array.from(input.data);
		setGrid(next);
		updatePrediction(next);
	};

	const clear = () => {
		const next = Array.from({ length: GRID_SIZE * GRID_SIZE }, () => 0);
		setGrid(next);
		updatePrediction(next);
	};

	const startTraining = () => {
		if (training) return;
		setTraining(true);
		setLossHistory([]);
		setEpochsDone(0);

		const rng = gridRng(42);
		const { inputs, targets } = generateGridSamples(rng, 200);

		const batchSize = 50;
		let currentEpoch = 0;

		const step = () => {
			const losses: number[] = [];
			train(network, inputs, targets, batchSize, 0.01, (_epoch, avgLoss) => {
				losses.push(avgLoss);
			});
			currentEpoch += batchSize;
			setEpochsDone(currentEpoch);
			setLossHistory((prev) => [...prev, ...losses]);
			onNetworkChange({ layers: network.layers.map((l) => ({ ...l })) });
			updatePrediction(grid);

			if (currentEpoch < 500) {
				requestAnimationFrame(step);
			} else {
				setTraining(false);
			}
		};

		requestAnimationFrame(step);
	};

	const pts = countPoints();
	const targetLabel = pts === 1 ? 0 : pts === 2 ? 1 : null;

	return (
		<div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5 mb-8">
			<h2 className="text-lg font-semibold mb-4">{t("grid.title")}</h2>

			<div className="flex flex-col md:flex-row gap-6">
				{/* Grid */}
				<div className="flex flex-col items-center gap-3">
					<div
						className="grid gap-1"
						style={{
							gridTemplateColumns: `repeat(${GRID_SIZE}, 1fr)`,
							width: "280px",
						}}
					>
						{(() => {
							const buttons: ReactNode[] = [];
							for (let r = 0; r < GRID_SIZE; r++) {
								for (let c = 0; c < GRID_SIZE; c++) {
									const idx = r * GRID_SIZE + c;
									buttons.push(
										<button
											key={`cell-${r}-${c}`}
											type="button"
											aria-label={`cell-${r}-${c}`}
											onClick={() => toggleCell(idx)}
											className={`aspect-square rounded transition ${
												grid[idx]
													? "bg-blue-600 shadow-inner"
													: "bg-slate-100 hover:bg-slate-200 border border-slate-200"
											}`}
										/>,
									);
								}
							}
							return buttons;
						})()}
					</div>
					<div className="text-xs text-slate-500">
						{t("grid.pointsCount", { count: pts })}
					</div>
				</div>

				{/* Controls & prediction */}
				<div className="flex-1 space-y-4">
					<div className="flex gap-2 flex-wrap">
						<button
							type="button"
							onClick={() => randomize(1)}
							className="px-3 py-1.5 bg-slate-100 text-slate-700 rounded font-medium hover:bg-slate-200 transition text-sm"
						>
							{t("grid.random1")}
						</button>
						<button
							type="button"
							onClick={() => randomize(2)}
							className="px-3 py-1.5 bg-slate-100 text-slate-700 rounded font-medium hover:bg-slate-200 transition text-sm"
						>
							{t("grid.random2")}
						</button>
						<button
							type="button"
							onClick={clear}
							className="px-3 py-1.5 bg-slate-100 text-slate-700 rounded font-medium hover:bg-slate-200 transition text-sm"
						>
							{t("grid.clear")}
						</button>
					</div>

					<div className="p-4 bg-slate-100 rounded-lg">
						<div className="text-sm text-slate-600 mb-1">
							{t("grid.prediction")}
						</div>
						<div className="text-2xl font-bold text-blue-700">
							{prediction.toFixed(4)}
						</div>
						{targetLabel !== null && (
							<div className="text-xs text-slate-500 mt-1">
								{t("grid.target", { value: targetLabel })}
							</div>
						)}
					</div>

					<button
						type="button"
						onClick={startTraining}
						disabled={training}
						className="w-full px-4 py-2.5 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
					>
						{training
							? t("grid.training", { count: epochsDone })
							: t("grid.train")}
					</button>

					{lossHistory.length > 0 && (
						<div className="text-xs text-slate-500">
							{t("grid.loss", {
								value: lossHistory[lossHistory.length - 1].toFixed(4),
							})}
						</div>
					)}
				</div>
			</div>
		</div>
	);
}
