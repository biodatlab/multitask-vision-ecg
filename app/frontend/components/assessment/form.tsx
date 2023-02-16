import {
  Box,
  Button,
  Checkbox,
  CheckboxGroup,
  Divider,
  Flex,
  FormControl,
  FormErrorMessage,
  FormLabel,
  forwardRef,
  Heading,
  HStack,
  Input,
  Radio,
  RadioGroup,
  RadioProps,
  StackDivider,
  Text,
  TextProps,
  VStack,
} from "@chakra-ui/react";
import React, { ReactNode, useCallback, useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import assessmentQuestions from "../../data/assessmentQuestions.json";
import type { prediction } from "../../pages/ecg";

// -- CONSTANTS
const thisYear = new Date().getFullYear() + 543;
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

// -- WRAPPED COMPONENTS
const WrappedRadio = forwardRef<
  {
    radioProps: RadioProps;
    customProps: { checked: boolean; error: any | undefined | null };
  },
  "div"
>(({ radioProps, customProps }, ref) => (
  <Box
    w={24}
    backgroundColor={customProps.checked ? "secondary.50" : undefined}
    borderRadius="xl"
    borderWidth={customProps.error ? "2px" : "1px"}
    borderStyle="solid"
    borderColor={customProps.error ? "red.500" : "gray.200"}
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

// Disease Emphasized Text
const DE = (props: TextProps) => <Text as="b" {...props} />;

// -- FUNCTION
const emphasizeDiseaseText = (
  rawText: string
): Array<ReactNode> | undefined => {
  if (!rawText) {
    return undefined;
  }

  /**
   * @see https://regexr.com/3fcqh
   */
  const boldTexts = Array.from(
    rawText.matchAll(/\*{2}((?=[^\s\*]).*?[^\s\*])\*{2}/g)
  );
  const positions = boldTexts.map(([raw, noSymbol]) => {
    const start = rawText.indexOf(raw);
    const end = start + raw.length;

    return { start, end, text: noSymbol };
  });

  let composed = [];

  if (positions.length === 0) {
    composed.push(rawText);
    return composed;
  }

  // construct text nodes
  let movingIndex = 0;

  for (let i = 0; i < positions.length; i++) {
    const { start, end, text } = positions[i];
    const textBefore = rawText.slice(movingIndex, start);

    composed.push(
      <React.Fragment key={textBefore + i}>{textBefore}</React.Fragment>
    );
    composed.push(<DE key={text + i}>{text}</DE>);
    movingIndex = end;

    if (i === positions.length - 1) {
      const textAfter = rawText.slice(end);
      composed.push(textAfter);
    }
  }

  return composed;
};

// -- MAIN
interface FormProps {
  onCalculate: () => void;
  onResult: (results: Array<prediction>) => void;
  isCalculating: boolean;
}

const Form = ({ onCalculate, onResult, isCalculating }: FormProps) => {
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

  // calculate here
  const onSubmit = useCallback(
    (data: { [key: string]: number }) => {
      // set isCalculating outside
      onCalculate();

      const selectedDiseases: Array<string> = watchDiseaseSelection;

      // fetch result from /api route
      fetch("/api/assessment", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          selectedDiseases,
          data,
        }),
      })
        .then((res) => res.json())
        .then((resJson) => {
          // mock processing behavior
          setTimeout(() => {
            onResult(resJson);
          }, 555);
        });
    },
    [onCalculate, onResult, watchDiseaseSelection]
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
      {/* disease selection block */}
      <Box
        w="100%"
        backgroundColor="white"
        borderRadius="2xl"
        boxShadow="lg"
        py={10}
        px={14}
        mb={6}
      >
        <FormControl>
          <FormLabel>
            <VStack mb={6} align="start">
              <Heading as="h6" fontSize="lg" color="secondary.400">
                แบบประเมินนี้สามารถประเมินภาวะโรคได้หลากหลาย
              </Heading>
              <Text as="b" fontSize="sm" color="secondary.400">
                กรุณาเลือกภาวะโรคที่ต้องการประเมิน
                โดยที่จำนวนภาวะโรคที่เลือกจะมีผลกับจำนวนข้อคำถามที่จะต้องตอบ
              </Text>
            </VStack>
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
      </Box>

      {/* question block */}
      {requiredQuestions?.length > 0 && (
        <Box
          w="100%"
          backgroundColor="white"
          borderRadius="2xl"
          boxShadow="lg"
          py={10}
          px={14}
        >
          <Box>
            <VStack mb={6} align="start">
              <Heading as="h6" fontSize="lg" color="grey.800">
                กรุณาตอบคำถามด้านล่างให้ครบ
              </Heading>
              <Text as="b" fontSize="sm" color="secondary.400">
                กดปุ่มคำนวณความเสี่ยงด้านล่างเพื่อรับผลการประเมินความเสี่ยง
              </Text>
            </VStack>

            <VStack gap={6} divider={<StackDivider borderColor="gray.200" />}>
              {requiredQuestions.map((value, ind) => {
                if (value.type === "input") {
                  // implement validation for year_of_birth
                  let option: { [index: string]: any } = {
                    required: !value.optional,
                  };

                  if (value.key === "year_of_birth") {
                    const thisYear = new Date().getFullYear() + 543;

                    option = {
                      ...option,
                      validate: (val: number) =>
                        val < thisYear && val > thisYear - 120,
                    };
                  }

                  return (
                    <FormControl key={value.key} isInvalid={errors[value.key]}>
                      <FormLabel htmlFor={value.key}>{`${ind + 1}. ${
                        value.text
                      }`}</FormLabel>
                      <Box pl={5}>
                        <Input
                          id={value.key}
                          type="number"
                          w={24}
                          {...register(value.key, option)}
                        />

                        {/* only displays validation error message for year_of_birth */}
                        {errors["year_of_birth"]?.type === "validate" && (
                          <FormErrorMessage>
                            {`ปีเกิดควรอยู่ในช่วง ${thisYear - 120} - ${
                              thisYear - 1
                            }`}
                          </FormErrorMessage>
                        )}
                      </Box>
                    </FormControl>
                  );
                }

                if (value.type === "radio") {
                  return (
                    <FormControl key={value.key} isInvalid={errors[value.key]}>
                      <FormLabel htmlFor={value.key}>
                        {`${ind + 1}. `}
                        {emphasizeDiseaseText(value.text)}
                      </FormLabel>
                      <RadioGroup pl={5}>
                        <HStack>
                          {value.choices.map((choice, ind) => {
                            return (
                              <WrappedRadio
                                key={choice}
                                customProps={{
                                  checked: watchAll[value.key] === String(ind),
                                  error: errors[value.key],
                                }}
                                radioProps={{
                                  children: choice,
                                  value: String(ind),
                                  ...register(value.key, {
                                    required: !value.optional,
                                  }),
                                }}
                              />
                            );
                          })}
                        </HStack>
                      </RadioGroup>
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
                isLoading={isSubmitting || isCalculating}
                type="submit"
              >
                คำนวณความเสี่ยง
              </Button>
            </Flex>
          </Box>
        </Box>
      )}
    </form>
  );
};

export default Form;
