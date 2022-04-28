import {
  Box,
  Button,
  Checkbox,
  CheckboxGroup,
  Divider,
  Flex,
  FormControl,
  FormLabel,
  forwardRef,
  Heading,
  HStack,
  Input,
  Radio,
  RadioGroup,
  RadioProps,
  StackDivider,
  VStack,
} from "@chakra-ui/react";
import { useCallback, useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import assessmentQuestions from "../../data/assessmentQuestions.json";
import type { prediction } from "../../pages/ecg";
import formula from "../../data/formula";

// -- CONSTANTS
const questionsMap: { [index: string]: number[] } = {
  scar: [2, 4, 7, 8, 9, 10, 11],
  cadScar: [2, 5, 7, 8, 9, 11],
  lvef40: [1, 2, 3, 4, 8, 9, 10, 11],
  lvef50: [1, 2, 3, 4, 6, 8, 9, 10],
};

const allQuestions: { [index: string]: any } = assessmentQuestions;
const defaultQuestions = Object.entries(assessmentQuestions).map(
  ([, value]) => value
);

// -- TYPES
type InferredQuestionType = Omit<typeof allQuestions["1"], "choices"> & {
  choices: never[] | string[];
};

// -- FUNCTIONS
const getRiskLabel = (percent: number) => {
  if (percent < 30) {
    return "ต่ำ";
  }

  if (percent >= 30 && percent < 70) {
    return "ปานกลาง";
  }

  return "สูง";
};
const snakeToCamel = (str: string): string =>
  str.toLowerCase().replace(/(_\w)/g, (m) => m.toUpperCase().substr(1));

// -- WRAPPED COMPONENTS
const WrappedRadio = forwardRef<
  { radioProps: RadioProps; customProps: { checked: boolean } },
  "div"
>(({ radioProps, customProps }, ref) => (
  <Box
    w={24}
    backgroundColor={customProps.checked ? "secondary.50" : undefined}
    borderRadius="xl"
    borderWidth="1px"
    borderStyle="solid"
    borderColor="gray.200"
    p={2}
    pl={3}
  >
    <Radio
      colorScheme="secondary"
      sx={{
        "&[aria-checked=true], &[data-checked]": {
          background: "secondary.400",
          borderColor: "secondary.400",
        },
      }}
      ref={ref}
      {...radioProps}
    />
  </Box>
));

// -- MAIN
interface FormProps {
  onCalculate: (results: Array<prediction>) => void;
}

const Form = ({ onCalculate }: FormProps) => {
  const {
    register,
    handleSubmit,
    watch,
    formState: { errors, isSubmitting },
  } = useForm();

  // to get required questions
  const watchDiseaseSelection = watch("diseases_selection");
  // use watch to help style radio buttons
  const watchAll = watch();

  /**
   * TODO:
   * - move calculate to /api
   * - fix type
   */
  // calculate here
  const onSubmit = useCallback(
    (data: any) => {
      const dataCamelCase = Object.entries(data).reduce(
        (acc, [key, value]) => ({
          ...acc,
          [snakeToCamel(key)]: value,
        }),
        {}
      );
      const selectedDiseases: Array<string> = watchDiseaseSelection;

      const results = selectedDiseases.reduce<Array<prediction>>((acc, cur) => {
        let logit;
        let title;
        let description;

        if (cur === "scar") {
          logit = formula.scar(dataCamelCase);
          title = "Myocardial Scar";
          description = "ความน่าจะเป็นที่จะมีแผลเป็นที่กล้ามเนื้อหัวใจ";
        }
        if (cur === "cadScar") {
          logit = formula.cadScar(dataCamelCase);
          title = "Coronary Artery Disease (CAD)";
          description = "ความน่าจะเป็นของการโรคหลอดแดงของหัวใจตีบหรือตัน";
        }
        if (cur === "lvef40") {
          logit = formula.lvef40(dataCamelCase);
          title = "LVEF < 40";
          description =
            "ความน่าจะเป็นที่ค่าประสิทธิภาพการทำงานของหัวใจห้องล่างซ้ายต่ำกว่า 40%";
        }
        if (cur === "lvef50") {
          logit = formula.lvef50(dataCamelCase);
          title = "LVEF < 50";
          description =
            "ความน่าจะเป็นที่ค่าประสิทธิภาพการทำงานของหัวใจห้องล่างซ้ายต่ำกว่า 50%";
        }

        const prob = formula.logitToProb(logit);

        return [
          ...acc,
          {
            title,
            description,
            risk_level: getRiskLabel(prob),
            probability: prob,
          },
        ];
      }, [] as Array<prediction>);

      onCalculate(results);
    },
    [onCalculate, watchDiseaseSelection]
  );

  // effect to get questions
  useEffect(() => {
    const selectedDiseases = watchDiseaseSelection;

    if (Array.isArray(selectedDiseases)) {
      const questionsSet = new Set<number>(
        selectedDiseases.reduce(
          (acc, cur) => [...acc, ...questionsMap[cur]],
          []
        )
      );
      const questionKeys = Array.from(questionsSet).sort((a, b) => a - b);
      const questions = questionKeys.map((key) => allQuestions[String(key)]);

      setRequiredQuestions(questions);
    }
  }, [watchDiseaseSelection]);

  const [requiredQuestions, setRequiredQuestions] =
    useState<InferredQuestionType[]>(defaultQuestions);

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      {/* disease selection */}
      <FormControl mb={6}>
        <FormLabel>
          <Heading as="h6" fontSize="md" color="secondary.400" mb={6}>
            กรุณาเลือกภาวะโรคที่ต้องการประเมินความเสี่ยง
          </Heading>
        </FormLabel>
        <CheckboxGroup
          colorScheme="secondary"
          defaultValue={["scar", "cadScar", "lvef40", "lvef50"]}
        >
          <VStack
            alignItems="flex-start"
            sx={{
              "& .chakra-checkbox__control[data-checked]": {
                backgroundColor: "secondary.400",
                borderColor: "secondary.400",
              },
            }}
          >
            <Checkbox value="scar" {...register("diseases_selection")}>
              รอยแผลเป็นในหัวใจ
            </Checkbox>
            <Checkbox value="cadScar" {...register("diseases_selection")}>
              หลอดเลือดแดงของหัวใจตีบหรือตัน
            </Checkbox>
            <Checkbox value="lvef40" {...register("diseases_selection")}>
              ประสิทธิภาพการบีบตัวของหัวใจห้องล่างซ้ายต่ำกว่า 40%
            </Checkbox>
            <Checkbox value="lvef50" {...register("diseases_selection")}>
              ประสิทธิภาพการบีบตัวของหัวใจห้องล่างซ้ายต่ำกว่า 50%
            </Checkbox>
          </VStack>
        </CheckboxGroup>
      </FormControl>

      {/* question to acquire */}
      {requiredQuestions?.length > 0 && (
        <Box>
          <Heading as="h6" fontSize="md" color="secondary.400" mb={6}>
            กรุณาตอบคำถามด้านล่างให้ครบ
            และกดปุ่มคำนวณความเสี่ยงด้านล่างเพื่อรับผลการประเมินความเสี่ยง
          </Heading>

          <VStack gap={6} divider={<StackDivider borderColor="gray.200" />}>
            {requiredQuestions.map((value, ind) => {
              if (value.type === "input") {
                return (
                  <FormControl
                    key={value.key}
                    isRequired={!value.optional}
                    isInvalid={errors[value.key]}
                  >
                    <FormLabel htmlFor={value.key}>{`${ind + 1}. ${
                      value.text
                    }`}</FormLabel>
                    <Box pl={5}>
                      <Input
                        id={value.key}
                        type="number"
                        w={24}
                        {...register(value.key)}
                      />
                    </Box>
                  </FormControl>
                );
              }

              if (value.type === "radio") {
                return (
                  <FormControl
                    key={value.key}
                    isRequired={!value.optional}
                    isInvalid={errors[value.key]}
                  >
                    <FormLabel htmlFor={value.key}>{`${ind + 1}. ${
                      value.text
                    }`}</FormLabel>
                    <RadioGroup pl={5}>
                      <HStack>
                        {value.choices.map((choice, ind) => {
                          return (
                            <WrappedRadio
                              key={choice}
                              customProps={{
                                checked: watchAll[value.key] === String(ind),
                              }}
                              radioProps={{
                                children: choice,
                                value: String(ind),
                                ...register(value.key),
                              }}
                            />
                          );
                        })}
                      </HStack>
                    </RadioGroup>
                    {/* <FormHelperText>We'll never share your email.</FormHelperText> */}
                  </FormControl>
                );
              }
            })}
          </VStack>
          <Divider borderColor="gray.200" mt={8} />
          <Flex justifyContent="flex-end" pt={2}>
            <Button
              mt={4}
              colorScheme="primary"
              backgroundColor="primary.300"
              isLoading={isSubmitting}
              type="submit"
            >
              คำนวณความเสี่ยง
            </Button>
          </Flex>
        </Box>
      )}
    </form>
  );
};

export default Form;