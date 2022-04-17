export type AssessmentProps = {
  year_of_birth: number;
  height: number;
  weight: number;
  female: number;
  diabetes_mellitus: number;
  history_mi: number;
  history_ptca_cabg: number;
  history_cad: number;
  history_hf: number;
  hypertension: number;
  history_ckd: number;
};

const scar = ({
  female,
  historyMi,
  historyPtcaCabg,
  historyCad,
  historyHf,
  historyCkd,
}: {
  [key: string]: number;
}) => {
  return (
    -1.25369 +
    -1.289421 * female +
    1.26799 * historyMi +
    0.852316 * historyPtcaCabg +
    1.188963 * historyCad +
    0.875971 * historyHf +
    0.394268 * historyCkd
  );
};

const cadScar = ({
  female,
  diabetesMellitus,
  historyMi,
  historyPtcaCabg,
  historyCad,
  historyHf,
  historyCkd,
}: {
  [key: string]: number;
}) => {
  return (
    -1.838088 +
    -1.257548 * female +
    0.290813 * diabetesMellitus +
    1.42843 * historyMi +
    0.915726 * historyPtcaCabg +
    1.477169 * historyCad +
    0.519559 * historyHf +
    0.35904 * historyCkd
  );
};

const lvef40 = ({
  yearOfBirth,
  female,
  height,
  weight,
  hypertension,
  historyCad,
  historyHf,
  historyCkd,
}: {
  [key: string]: number;
}) => {
  const thisYear = new Date().getFullYear();
  const ageAbove65: number = thisYear - (yearOfBirth - 543) >= 65 ? 1 : 0;
  const heightInMetre = height / 100;
  const bmiAbove25: number =
    weight / (heightInMetre * heightInMetre) > 25 ? 1 : 0;

  return (
    -1.986239 +
    -0.573404 * ageAbove65 +
    -0.537664 * female +
    -0.771137 * bmiAbove25 +
    -0.533213 * hypertension +
    0.714925 * historyCad +
    2.065291 * historyHf +
    0.61929 * historyCkd
  );
};

const lvef50 = ({
  yearOfBirth,
  female,
  height,
  weight,
  hypertension,
  historyMi,
  historyCad,
  historyHf,
}: {
  [key: string]: number;
}) => {
  const thisYear = new Date().getFullYear();
  const ageAbove65: number = thisYear - (yearOfBirth - 543) >= 65 ? 1 : 0;
  const heightInMetre = height / 100;
  const bmiAbove25: number =
    weight / (heightInMetre * heightInMetre) > 25 ? 1 : 0;

  return (
    -1.468317 +
    -0.438355 * ageAbove65 +
    -0.797398 * female +
    -0.3547618 * bmiAbove25 +
    -0.350557 * hypertension +
    0.534031 * historyMi +
    0.787748 * historyCad +
    1.831592 * historyHf
  );
};

const logitToProb = (logit: number) => {
  return Math.round((1 / (1 + Math.exp(-logit))) * 100);
};

const formula = { scar, cadScar, lvef40, lvef50, logitToProb };

export default formula;
