import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { StepIndicator } from "./StepIndicator";
import { STEPS } from "../types/steps";

describe("StepIndicator", () => {
  it("renders all seven steps", () => {
    render(<StepIndicator currentStep="upload" maxStepIndexReached={0} onNavigate={() => {}} />);
    expect(screen.getAllByRole("listitem")).toHaveLength(7);
    for (const step of STEPS) {
      expect(screen.getByText(step.label)).toBeInTheDocument();
    }
  });

  it("marks the current step and only the current step", () => {
    render(<StepIndicator currentStep="training" maxStepIndexReached={3} onNavigate={() => {}} />);
    const current = screen.getAllByRole("button", { current: "step" });
    expect(current).toHaveLength(1);
    // Training is the 4th step.
    expect(current[0]).toHaveTextContent("4");
  });

  it("marks nothing current when no step is null (the Start screen)", () => {
    const { container } = render(
      <StepIndicator currentStep={null} maxStepIndexReached={-1} onNavigate={() => {}} />,
    );
    expect(container.querySelectorAll('[aria-current="step"]')).toHaveLength(0);
  });

  it("jumps to a completed step on click, but not the current or a future one", async () => {
    const user = userEvent.setup();
    const onNavigate = vi.fn();
    // On "model-selection" (index 2), having reached index 3 ("training").
    render(
      <StepIndicator currentStep="model-selection" maxStepIndexReached={3} onNavigate={onNavigate} />,
    );

    await user.click(screen.getByRole("button", { name: "1" })); // completed (Upload)
    expect(onNavigate).toHaveBeenCalledWith(0);

    await user.click(screen.getByRole("button", { name: "4" })); // completed (Training)
    expect(onNavigate).toHaveBeenCalledWith(3);

    onNavigate.mockClear();
    await user.click(screen.getByRole("button", { name: "3" })); // current -- disabled, no-op
    expect(onNavigate).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "5" })); // future -- disabled, no-op
    expect(onNavigate).not.toHaveBeenCalled();
  });

  it("disables every step when nothing has been completed yet", () => {
    render(<StepIndicator currentStep={null} maxStepIndexReached={-1} onNavigate={() => {}} />);
    for (const button of screen.getAllByRole("button")) {
      expect(button).toBeDisabled();
    }
  });
});
