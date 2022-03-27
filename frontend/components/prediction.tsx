import {
  Box,
  Flex,
  Heading,
  Stack,
  Switch,
  Text,
  useDisclosure,
} from "@chakra-ui/react";

interface PredictionProps {
  normality: number;
  lvefgteq40: number;
  lveflw50: number;
}

interface ValueBoxProps {
  title: string;
  labelLt: string;
  labelRt: string;
  value: number;
}

const ValueBox = ({ title, labelLt, labelRt, value }: ValueBoxProps) => {
  const valueLt = value.toFixed(2);
  const valueRt = (1 - value).toFixed(2);
  const leftOffset = (value * 100).toFixed(2);

  return (
    <Flex flex={1} flexDirection="column" alignItems="center" px={10}>
      <Heading size="md" as="h3" color="gray.600" textAlign="center" mb={4}>
        {title}
      </Heading>
      <Stack direction="row" w="100%" gap={1} alignItems="flex-start">
        <Box>
          <Text fontSize="xs" textAlign="center">
            <Text as="span" fontSize="125%" fontWeight="bold">
              {valueLt}
            </Text>
            <br />
            {labelLt}
          </Text>
        </Box>
        <Flex flex={1} pt={1} position="relative">
          <Box h={4} w="100%" borderRadius="xl" backgroundColor="pink.200" />
          <Box
            h={6}
            w="2px"
            position="absolute"
            top={0}
            left={`${leftOffset}%`}
            backgroundColor="gray.500"
            borderRadius="lg"
          />
        </Flex>
        <Box>
          <Text fontSize="xs" textAlign="center">
            <Text as="span" fontSize="125%" fontWeight="bold">
              {valueRt}
            </Text>
            <br />
            {labelRt}
          </Text>
        </Box>
      </Stack>
    </Flex>
  );
};

const Prediction = ({ normality, lvefgteq40, lveflw50 }: PredictionProps) => {
  const {
    isOpen: isOpenPanel,
    onClose: onClosePanel,
    onOpen: onOpenPanel,
  } = useDisclosure();

  return (
    <Stack direction="column" gap={4}>
      <Flex justifyContent="flex-end" alignItems="center">
        <Box>
          <Switch
            size="md"
            onChange={(e) =>
              e.currentTarget.checked ? onOpenPanel() : onClosePanel()
            }
          />
          &nbsp; เปิด / ปิด ความน่าจะเป็น
        </Box>
      </Flex>
      {isOpenPanel && (
        <Stack direction="row">
          <ValueBox
            title="ปกติ"
            labelLt="ปกติ"
            labelRt="ผิดปกติ"
            value={normality}
          />
          <ValueBox
            title="LVEF >= 40"
            labelLt=">= 40"
            labelRt="< 40"
            value={lvefgteq40}
          />
          <ValueBox
            title="LVEF < 50"
            labelLt=">= 50"
            labelRt="< 50>"
            value={lveflw50}
          />
        </Stack>
      )}
    </Stack>
  );
};

export default Prediction;
