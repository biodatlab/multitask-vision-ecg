import {
  Box,
  Button,
  Container,
  Flex,
  Heading,
  Input,
  Radio,
  RadioGroup,
  Stack,
  Table,
  TableCaption,
  TableContainer,
  Tbody,
  Td,
  Text,
  Tfoot,
  Th,
  Thead,
  Tr,
  VStack,
} from "@chakra-ui/react";
import { useState } from "react";
import Bar from "../components/bar";
import Layout from "../components/layout";

interface Answers {
  age?: number;
  sex?: number;
  history_heart_failure?: number;
  history_cad?: number;
  implantable?: number;
  diabetes?: number;
  hypertension?: number;
  smoking?: number;
  renal_replacement?: number;
  lvef?: number;
}

interface Question {
  key:
    | "age"
    | "sex"
    | "history_heart_failure"
    | "history_cad"
    | "implantable"
    | "diabetes"
    | "hypertension"
    | "smoking"
    | "renal_replacement"
    | "lvef";
  text: string;
  type: string;
  optional: boolean;
  choices: Array<string>;
}

// all inputs are number
const questions: Array<Question> = [
  {
    key: "age",
    text: "อายุ (ปี)",
    type: "input",
    optional: false,
    choices: [],
  },
  {
    key: "sex",
    text: "เพศ",
    type: "radio",
    optional: false,
    choices: ["ชาย", "หญิง"],
  },
  {
    key: "history_heart_failure",
    text: "มีประวัติโรคหัวใจล้มเหลว",
    type: "radio",
    optional: false,
    choices: ["ไม่มี", "มี"],
  },
  {
    key: "history_cad",
    text: "มีประวัติโรคกล้ามเนื้อหัวใจขาดเลือด",
    type: "radio",
    optional: false,
    choices: ["ไม่มี", "มี"],
  },
  {
    key: "implantable",
    text: "มีอุปกรณ์ฝังช่วย",
    type: "radio",
    optional: false,
    choices: ["ไม่มี", "มี"],
  },
  {
    key: "diabetes",
    text: "เป็นโรคเบาหวาน",
    type: "radio",
    optional: false,
    choices: ["ไม่ใช่", "ใช่"],
  },
  {
    key: "hypertension",
    text: "ความดันสูง",
    type: "radio",
    optional: false,
    choices: ["ไม่ใช่", "ใช่"],
  },
  {
    key: "smoking",
    text: "เคยสูบบุหรี่",
    type: "radio",
    optional: false,
    choices: ["ไม่ใช่", "ใช่"],
  },
  {
    key: "renal_replacement",
    text: "เคยฟอกไต",
    type: "radio",
    optional: false,
    choices: ["ไม่ใช่", "ใช่"],
  },
  {
    key: "lvef",
    text: "ค่าประสิทธิภาพหัวใจห้องล่างซ้าย (ไม่บังคับ)",
    type: "input",
    optional: true,
    choices: [],
  },
];

const calculate = (
  answers: Answers,
  onInvalid: (invalidFields: Array<string>) => void
): number | null => {
  const {
    age,
    sex,
    history_heart_failure,
    history_cad,
    implantable,
    diabetes,
    hypertension,
    smoking,
    renal_replacement,
    lvef,
  } = answers;

  if (
    age === undefined ||
    sex === undefined ||
    history_heart_failure === undefined ||
    history_cad === undefined ||
    implantable === undefined ||
    diabetes === undefined ||
    hypertension === undefined ||
    smoking === undefined ||
    renal_replacement === undefined
  ) {
    const requiredFields = questions
      .map((x) => x.key)
      .filter((x) => x !== "lvef");

    const invalidFields = requiredFields.filter((k) => {
      return answers[k] === undefined;
    });

    onInvalid(invalidFields);

    return null;
  }

  let prognostic_index;
  let prediction;

  if (!lvef) {
    prognostic_index =
      0.02 * age +
      0.452 * sex +
      1.019 * history_heart_failure +
      0.579 * history_cad +
      0.453 * implantable +
      0.704 * diabetes +
      0.297 * hypertension +
      0.506 * smoking +
      1.508 * renal_replacement;
    prediction = 1 - 0.99535167 ** Math.exp(prognostic_index);
  } else {
    prognostic_index =
      0.025 * age +
      0.563 * sex +
      0.772 * history_heart_failure +
      0.463 * history_cad +
      0.413 * implantable +
      0.732 * diabetes +
      0.356 * hypertension +
      0.523 * smoking +
      1.516 * renal_replacement -
      0.021 * lvef;
    prediction = 1 - 0.98801461 ** Math.exp(prognostic_index);
  }

  return prediction * 100;
};

const Manual = () => {
  const [answers, setAnswers] = useState({} as Answers);
  const [invalidFields, setInvalidFields] = useState([] as Array<string>);

  // With LVEF available:
  // prognostic_index = (0.025 * age) + (0.563 * female) + (0.772 * history_heart_failure) + (0.463 * history_cad) + (0.413 * implantable) + (0.732  * diabetes) + (0.356 * hypertension) + (0.523 * smoking) + (1.516 * renal_replacement) - (0.021 * lvef)
  // prediction = 1 - 0.98801461 ** (exp(prognostic_index))

  // Without LVEF:
  // prognostic_index = (0.020 * age) + (0.452 * female) + (1.019 * history_heart_failure) + (0.579 * history_cad) + (0.453 * implantable) + (0.704  * diabetes) + (0.297 * hypertension) + (0.506 * smoking) + (1.508 * renal_replacement)
  // prediction = 1 - 0.99535167 ** (exp(prognostic_index))

  return (
    <Layout>
      <Stack direction="column" gap={2} py={6}>
        <Heading as="h1">
          แบบประเมินความเสี่ยงโรคหัวใจล้มเหลวภายในระยะเวลา 3 ปี
        </Heading>
        <Flex justifyContent="center">
          <VStack gap={4}>
            <Container maxW="container.md" centerContent>
              <TableContainer>
                <Table variant="simple">
                  <Thead>
                    <Tr>
                      <Th>คำถาม</Th>
                      <Th>คำตอบ</Th>
                    </Tr>
                  </Thead>
                  <Tbody>
                    {questions.map((question) => (
                      <Tr
                        key={question.text}
                        background={
                          invalidFields.includes(question.key)
                            ? "red.100"
                            : undefined
                        }
                      >
                        <Td>{question.text}</Td>
                        <Td>
                          {question.type === "radio" ? (
                            <RadioGroup
                              name={question.key}
                              onChange={(val) => {
                                setAnswers((prev) => ({
                                  ...prev,
                                  [question.key]: val,
                                }));
                                // remove from invalid fields
                                setInvalidFields((prev) =>
                                  prev.filter((x) => x !== question.key)
                                );
                              }}
                              value={answers[question.key]}
                            >
                              <Stack direction="row">
                                {question.choices.map((choice, ind) => (
                                  <Radio
                                    key={choice}
                                    value={String(ind)}
                                    isRequired={!question.optional}
                                    isInvalid={invalidFields.includes(
                                      question.key
                                    )}
                                  >
                                    {choice}
                                  </Radio>
                                ))}
                              </Stack>
                            </RadioGroup>
                          ) : (
                            <Input
                              size="sm"
                              type="number"
                              value={answers[question.key] || ""}
                              isRequired={!question.optional}
                              isInvalid={invalidFields.includes(question.key)}
                              onChange={(e) => {
                                setAnswers((prev) => ({
                                  ...prev,
                                  [question.key]: Number(e.target.value),
                                }));
                                // remove from invalid fields
                                setInvalidFields((prev) =>
                                  prev.filter((x) => x !== question.key)
                                );
                              }}
                            />
                          )}
                        </Td>
                      </Tr>
                    ))}
                  </Tbody>
                </Table>
              </TableContainer>
              {invalidFields.length > 0 && (
                <Box mt={4}>
                  <Text fontWeight="bold" color="red.500">
                    กรุณาเลือก{" "}
                    {invalidFields
                      .map(
                        (x) => `"${questions.find((y) => y.key === x)?.text}"`
                      )
                      .join(", ")}
                  </Text>
                </Box>
              )}
            </Container>

            <Button
              onClick={() => {
                const pred = calculate(answers, (invalidFields) => {
                  setInvalidFields(invalidFields);
                });

                console.log({ pred });
              }}
              colorScheme="pink"
              px={6}
            >
              คำนวณ
            </Button>

            <Box w="100%" p={10} textAlign="center">
              <Text fontSize="3xl" fontWeight="bold">
                15%
              </Text>
              <Text fontSize="xl">ความเสี่ยงสูงมาก</Text>
              <Bar value={20} min={0} max={100} />
              <Button variant="ghost" colorScheme="pink">
                ดูสูตรคำนวณความเสี่ยง
              </Button>
            </Box>
          </VStack>
        </Flex>
      </Stack>
    </Layout>
  );
};

export default Manual;
