import { useCallback, useEffect, useState } from "react";
import { ScreenPanel } from "./components/ScreenPanel";
import { StepShell } from "./components/StepShell";
import { CompareScreen } from "./components/screens/CompareScreen";
import { EdaScreen } from "./components/screens/EdaScreen";
import { ModelSelectionScreen } from "./components/screens/ModelSelectionScreen";
import { PredictScreen } from "./components/screens/PredictScreen";
import { ResultsScreen } from "./components/screens/ResultsScreen";
import { StartScreen } from "./components/screens/StartScreen";
import { TrainingScreen } from "./components/screens/TrainingScreen";
import { UploadScreen } from "./components/screens/UploadScreen";
import { useModels } from "./hooks/useModels";
import { useTraining } from "./hooks/useTraining";
import type { DataProfileResponse } from "./hooks/useDataset";
import { STEPS, type StepId, type View } from "./types/steps";

const DEFAULT_TEST_SIZE = 0.2;

function NoDatasetNotice() {
  return (
    <ScreenPanel>
      <p className="text-sm text-muted">
        Upload a dataset or load a sample first.
      </p>
    </ScreenPanel>
  );
}

function App() {
  const [view, setView] = useState<View>("start");

  const [maxStepIndexReached, setMaxStepIndexReached] = useState(-1);

  const [profile, setProfile] = useState<DataProfileResponse | null>(null);
  const [testSize, setTestSize] = useState(DEFAULT_TEST_SIZE);
  const [selectedModelKeys, setSelectedModelKeys] = useState<string[]>([]);

  const modelsState = useModels();
  const trainingState = useTraining();

  const { checkCompatibility } = modelsState;
  const { reset: resetTraining } = trainingState;

  const currentStepIndex =
    view === "start" ? -1 : STEPS.findIndex((s) => s.id === view);

  const goToStepIndex = useCallback((index: number) => {
    setView(STEPS[index].id);
    setMaxStepIndexReached((m) => Math.max(m, index));
  }, []);

  const handleProfile = useCallback((next: DataProfileResponse) => {
    setProfile(next);
    setMaxStepIndexReached(0);
  }, []);

  useEffect(() => {
    if (!profile) return;

    setSelectedModelKeys([]);
    resetTraining();
    checkCompatibility(profile.data_id);
  }, [profile, checkCompatibility, resetTraining]);

  const toggleModel = useCallback(
    (key: string) => {
      setSelectedModelKeys((prev) =>
        prev.includes(key)
          ? prev.filter((k) => k !== key)
          : [...prev, key],
      );

      resetTraining();

      setMaxStepIndexReached((m) =>
        Math.min(
          m,
          STEPS.findIndex((s) => s.id === "model-selection"),
        ),
      );
    },
    [resetTraining],
  );

  const isStepComplete: Partial<Record<StepId, boolean>> = {
    upload: profile !== null,
    eda: profile !== null,
    "model-selection": selectedModelKeys.length > 0,
    training: trainingState.results !== null,
  };

  const currentStep = view === "start" ? null : (view as StepId);

  const currentComplete = currentStep
    ? (isStepComplete[currentStep] ?? true)
    : false;

  const canGoBack = currentStepIndex > 0;

  const canGoForward =
    currentStepIndex >= 0 &&
    currentStepIndex < STEPS.length - 1 &&
    currentComplete;

  /*
   * START SCREEN
   *
   * Render it outside StepShell completely.
   * This guarantees there is NO top bar or bottom navigation.
   */
  if (view === "start") {
    return (
      <StartScreen
        onBegin={() => {
          goToStepIndex(0);
        }}
      />
    );
  }

  /*
   * WORKFLOW
   *
   * StepShell is only rendered after Begin is clicked.
   */
  return (
    <StepShell
      view={view}
      maxStepIndexReached={maxStepIndexReached}
      onNavigateToStep={goToStepIndex}
      canGoBack={canGoBack}
      canGoForward={canGoForward}
      onBack={() =>
        setView(
          STEPS[Math.max(0, currentStepIndex - 1)].id,
        )
      }
      onForward={() =>
        goToStepIndex(
          Math.min(
            STEPS.length - 1,
            currentStepIndex + 1,
          ),
        )
      }
      renderStart={() => null}
      renderStep={(step) => {
        switch (step) {
          case "upload":
            return (
              <UploadScreen
                profile={profile}
                onProfile={handleProfile}
                testSize={testSize}
                onTestSizeChange={setTestSize}
              />
            );

          case "eda":
            return profile ? (
              <EdaScreen profile={profile} />
            ) : (
              <NoDatasetNotice />
            );

          case "model-selection":
            return profile ? (
              <ModelSelectionScreen
                models={modelsState.models}
                compatibility={modelsState.compatibility}
                loading={modelsState.loading}
                error={modelsState.error}
                selected={selectedModelKeys}
                onToggle={toggleModel}
              />
            ) : (
              <NoDatasetNotice />
            );

          case "training":
            return profile ? (
              <TrainingScreen
                dataId={profile.data_id}
                models={modelsState.models}
                selectedModelKeys={selectedModelKeys}
                testSize={testSize}
                onTestSizeChange={setTestSize}
                trainingState={trainingState}
              />
            ) : (
              <NoDatasetNotice />
            );

          case "results":
            return profile ? (
              <ResultsScreen results={trainingState.results} />
            ) : (
              <NoDatasetNotice />
            );

          case "predict":
            return profile ? (
              <PredictScreen
                profile={profile}
                trainingResults={trainingState.results}
              />
            ) : (
              <NoDatasetNotice />
            );

          case "compare":
            return profile ? (
              <CompareScreen
                trainingResults={trainingState.results}
              />
            ) : (
              <NoDatasetNotice />
            );

          default:
            return <NoDatasetNotice />;
        }
      }}
    />
  );
}

export default App;